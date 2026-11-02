# core/llm_log_sink.py —— 调用流水落库（异步、批量、不阻塞主流程）
#
# 为什么不能直接 await 写库：
#   一次生成有 20+ 次模型调用，若每次都在主流程里同步写一次 SQLite，
#   每次都多一次事务提交（实测毫秒级），并发写作时还会和业务写入抢写锁。
#   P0-2 刚把章节写作做成并发，这里再同步落库就把收益抵掉了一半。
#
# 做法：sink 只做一次「非阻塞入队」（put_nowait），后台协程每 0.5s 或攒够
# 50 条批量提交一次。主流程付出的代价是微秒级的队列操作。
#
# 丢数据的边界（明确写出来，不假装可靠）：
#   - 队列满（默认 1 万条）→ 丢弃并计数 dropped_full
#   - 没有运行中的事件循环（同步调用场景）→ 丢弃并计数 dropped_no_loop
#   - 落库异常 → 丢弃该批并计数 flush_failures，绝不重试阻塞后续
#   这三个计数就是「有没有丢」的唯一证据，第 3 步的接口会把它暴露出来。
import asyncio
import logging
import time
from typing import List, Optional

from app.core.llm_client import UsageRecord, get_bound_loop, register_sink
from app.database import AsyncSessionLocal
from app.models.llm_call_log import LlmCallLog

logger = logging.getLogger("app.llm.observability")

# 停止哨兵：走队列而不是 task.cancel()。
# 好处：① task 立刻被唤醒，stop() 不用死等轮询间隔；② 哨兵排在积压数据后面，
# 保证"先写完剩余的，再退出"，不会像 cancel 那样打断正在提交的事务。
_STOP = object()


class UsageLogSink:
    """把 UsageRecord 批量写进 llm_call_log 表。可作为 sink 直接注册。"""

    def __init__(self, batch_size: int = 50, flush_interval: float = 0.5, max_queue: int = 10000):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_queue = max_queue
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._task: Optional[asyncio.Task] = None
        # 启动时绑定的事件循环：回调可能不在 loop 线程里（见 __call__ 注释）
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # 统计口径
        self.received = 0
        self.written = 0
        self.flushes = 0
        self.dropped_full = 0
        self.dropped_no_loop = 0
        self.dropped_flush = 0
        self.flush_failures = 0

    # ---------------- 入口：必须是同步非阻塞 ----------------
    def start(self) -> bool:
        """绑定当前事件循环并启动后台 flush。在 lifespan 里显式调用一次。

        显式 start 的意义：把 loop 记下来，供跨线程回调投递使用。
        """
        loop = self._running_loop()
        if loop is None:
            return False
        self._loop = loop
        self._ensure_task(loop)
        return True

    def __call__(self, rec: UsageRecord) -> None:
        self.received += 1

        loop = self._running_loop()
        if loop is not None:
            self._loop = loop
            self._ensure_task(loop)
            self._put(rec)
            return

        # 这里没有 running loop，两种情况：
        #   ① 回调在**线程池**里执行（实测就是这条路径：langchain 0.3.63 即使
        #      回调是 AsyncCallbackHandler，ainvoke 下仍调同步 on_llm_end，线程 asyncio_0）
        #   ② 业务在线程里直接调模型
        # 直接判空丢弃会导致整批记录静默消失（表里 0 行、日志全正常），
        # 所以回退到「事件循环线程里记下的 loop」做线程安全投递。
        target = self._loop or get_bound_loop()
        if target is None or target.is_closed():
            self.dropped_no_loop += 1
            return
        try:
            target.call_soon_threadsafe(self._put, rec)
        except RuntimeError:        # loop 正在关闭
            self.dropped_no_loop += 1

    def _put(self, rec: UsageRecord) -> None:
        try:
            self._queue.put_nowait(rec)
        except asyncio.QueueFull:
            self.dropped_full += 1

    @staticmethod
    def _running_loop():
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    def _ensure_task(self, loop) -> None:
        # 懒启动：第一次有记录时自动起飞，调用方不用记得 start()。
        # 同时校验 loop：测试里每个用例是独立 loop，旧 task 已随旧 loop 失效。
        if self._task is not None and not self._task.done() and self._task.get_loop() is loop:
            return
        self._task = loop.create_task(self._run(), name="llm-log-flush")

    # ---------------- 后台批量提交 ----------------
    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            if first is _STOP:
                return

            batch: List[UsageRecord] = [first]
            deadline = time.monotonic() + self.flush_interval
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if nxt is _STOP:
                    # 先把手上的写完再退，避免丢掉这一批
                    await self._flush(batch)
                    return
                batch.append(nxt)
            await self._flush(batch)

    async def _flush(self, batch: List[UsageRecord]) -> None:
        if not batch:
            return
        try:
            async with AsyncSessionLocal() as db:
                db.add_all([self._to_row(r) for r in batch])
                await db.commit()
            self.written += len(batch)
            self.flushes += 1
        except Exception as e:  # noqa: BLE001 —— 观测写失败不能影响业务，也不能无限重试
            self.flush_failures += 1
            self.dropped_flush += len(batch)
            logger.warning("llm_call_log 落库失败，丢弃 %d 条：%s", len(batch), e)

    @staticmethod
    def _to_row(rec: UsageRecord) -> LlmCallLog:
        return LlmCallLog(
            paper_id=rec.paper_id,
            user_id=rec.user_id,
            node=rec.node,
            model=rec.model,
            prompt_tokens=rec.prompt_tokens,
            completion_tokens=rec.completion_tokens,
            cache_hit_tokens=rec.cache_hit_tokens,
            total_tokens=rec.total_tokens,
            cost=rec.cost,
            latency_ms=rec.latency_ms,
            status=rec.status,
            peak=rec.peak,
            error=(rec.error or None),
            created_at=rec.created_at,
        )

    # ---------------- 关闭：排空而不是 cancel ----------------
    async def stop(self) -> None:
        """停止后台任务并把队列里剩下的写完。

        不用 task.cancel()：如果此刻正在 _flush，取消会打断事务提交，
        那一批就静默丢了。改为投递哨兵，让后台协程自己走完收尾。
        """
        if self._task is not None and not self._task.done():
            try:
                self._queue.put_nowait(_STOP)
            except asyncio.QueueFull:
                pass          # 队列满说明还在消化，下面的 wait_for 兜底
            try:
                # 先让 pending 的跨线程投递（call_soon_threadsafe）落进队列，
                # 否则它们会在哨兵之后才到，被 stop 漏掉。
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                await asyncio.wait_for(asyncio.shield(self._task), timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        self._task = None

        # 兜底：哨兵之前的积压应由 _run 写完，这里再收一次尾（例如 sink 从未启动）
        rest: List[UsageRecord] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not _STOP:
                rest.append(item)
        if rest:
            await self._flush(rest)

    def stats(self) -> dict:
        """落库自检数据：收到 / 写入 / 丢弃，三者必须能对上。

        balanced 是这套异步方案的验收线 —— 如果它长期为 False，说明有记录在
        某条路径上凭空消失了，此时成本表就不可信。
        """
        accounted = (
            self.written + self.dropped_full + self.dropped_no_loop
            + self.dropped_flush + self._queue.qsize()
        )
        return {
            "received": self.received,
            "written": self.written,
            "flushes": self.flushes,
            "dropped_full": self.dropped_full,
            "dropped_no_loop": self.dropped_no_loop,
            "dropped_flush": self.dropped_flush,
            "flush_failures": self.flush_failures,
            "pending": self._queue.qsize(),
            "balanced": self.received == accounted,
        }


# 模块级单例：导入即注册（main.py 导入本模块）
usage_log_sink = UsageLogSink()
register_sink(usage_log_sink)

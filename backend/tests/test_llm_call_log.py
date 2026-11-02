# test_llm_call_log.py —— 调用流水落库：非阻塞、不丢数、失败不拖垮业务
#
# 这一层最危险的失败模式是"静默丢数据"：程序照跑，成本表少几行，没人发现。
# 所以测试重点不是"能写入"，而是三个边界：
#   ① 入队是否真的非阻塞（会不会把 P0-2 的并发收益吃掉）
#   ② stop 时队列残留是否写完（进程退出那一下最容易丢）
#   ③ 写库失败时是否计入丢弃（而不是假装写成功）
import asyncio
import time
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.core import llm_client
from app.core.llm_client import UsageRecord
from app.core.llm_log_sink import UsageLogSink
from app.database import AsyncSessionLocal
from app.models.llm_call_log import LlmCallLog


def _rec(node="write_sections", paper_id=1, tokens=(100, 50), cost=Decimal("0.0012")):
    return UsageRecord(
        node=node,
        model="deepseek-flash",
        prompt_tokens=tokens[0],
        completion_tokens=tokens[1],
        cache_hit_tokens=0,
        cost=cost,
        latency_ms=123.4,
        paper_id=paper_id,
    )


async def _rows():
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(LlmCallLog).order_by(LlmCallLog.id))
        return res.scalars().all()


# ---------------------------------------------------------------------------
# 表结构
# ---------------------------------------------------------------------------
def test_table_registered_in_metadata():
    """表必须注册进 Base.metadata，否则 create_all 不会建表（启动后第一次写就炸）。"""
    assert "llm_call_log" in LlmCallLog.metadata.tables
    cols = set(LlmCallLog.__table__.columns.keys())
    # 需求里点名的字段一个都不能少
    assert {
        "paper_id", "node", "model", "prompt_tokens", "completion_tokens",
        "cost", "latency_ms", "status",
    } <= cols


async def test_row_persisted_with_full_fields():
    """落库后每个字段都要能还原，尤其 cost 不能因为类型转换变成 0。"""
    sink = UsageLogSink(batch_size=10, flush_interval=0.01)
    sink(_rec(node="quality_check", paper_id=7, tokens=(1000, 500), cost=Decimal("0.01234567")))
    await sink.stop()

    rows = await _rows()
    assert len(rows) == 1
    row = rows[0]
    assert row.node == "quality_check"
    assert row.paper_id == 7
    assert row.prompt_tokens == 1000
    assert row.completion_tokens == 500
    assert row.total_tokens == 1500
    assert Decimal(str(row.cost)) == Decimal("0.01234567")
    assert row.status == "ok"
    assert row.created_at is not None


# ---------------------------------------------------------------------------
# 非阻塞：这是不拖累 P0-2 并发收益的前提
# ---------------------------------------------------------------------------
async def test_enqueue_is_non_blocking():
    """入队必须是微秒级。如果这里阻塞，每次模型调用都多一次磁盘写，并发收益被吃光。"""
    sink = UsageLogSink(batch_size=5000, flush_interval=30, max_queue=20000)
    recs = [_rec() for _ in range(2000)]

    t0 = time.perf_counter()
    for r in recs:
        sink(r)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.3, f"2000 次入队耗时 {elapsed:.3f}s，说明入队路径被阻塞了"
    assert elapsed / len(recs) < 0.0002, "单次入队超过 200 微秒，不符合非阻塞预期"
    await sink.stop()


async def test_batch_write_reduces_transactions():
    """批量提交：120 条记录不能产生 120 次事务，否则写入开销比调用本身还贵。"""
    sink = UsageLogSink(batch_size=50, flush_interval=0.05)
    for _ in range(120):
        sink(_rec())
    await sink.stop()

    assert sink.written == 120
    assert sink.flushes < 10, f"120 条用了 {sink.flushes} 次事务，批量没生效"
    assert len(await _rows()) == 120


# ---------------------------------------------------------------------------
# 不丢数
# ---------------------------------------------------------------------------
async def test_stop_drains_pending_queue():
    """进程退出（stop）时队列里还没落库的记录必须写完。

    这是异步落库最容易丢的一段：批量提交有窗口，退出瞬间不排空就少最后一批。
    """
    sink = UsageLogSink(batch_size=100, flush_interval=30)   # 几乎不会自动 flush
    for _ in range(5):
        sink(_rec())
    assert sink._queue.qsize() > 0          # 确认确实有积压

    await sink.stop()
    assert len(await _rows()) == 5
    assert sink.stats()["pending"] == 0


async def test_queue_full_drops_and_counts(monkeypatch):
    """队列满时只能丢，但必须计数 —— 静默丢会让成本表的"总计"偏低。"""
    sink = UsageLogSink(max_queue=5)
    monkeypatch.setattr(sink, "_ensure_task", lambda loop: None)   # 冻结消费端

    for _ in range(9):
        sink(_rec())
    assert sink.dropped_full == 4
    assert sink.stats()["balanced"] is True


def test_no_running_loop_drops_and_counts(monkeypatch):
    """没有运行中的事件循环、也没有已绑定的 loop 时，不能抛异常，只能计数丢弃。"""
    # 显式清掉可能残留的 loop 引用：否则上一个用例留下的 loop 会让这条记录被投递走，
    # 用例变成"取决于执行顺序"
    monkeypatch.setattr(llm_client, "_loop_ref", None)
    sink = UsageLogSink()
    sink(_rec())
    assert sink.dropped_no_loop == 1
    assert sink.stats()["balanced"] is True


async def test_cross_thread_callback_still_lands():
    """回归：回调在线程池线程里执行时，靠 call_soon_threadsafe 投递，不能整批丢弃。

    真实踩过的坑：langchain 在 ainvoke 时会把「同步回调」丢进线程池
    （实测线程名 asyncio_0，get_running_loop() 直接抛错），早期实现按"无 loop"
    把这批记录全丢了 —— 表里 0 行，但日志一切正常，属于最难发现的静默故障。
    """
    sink = UsageLogSink(batch_size=10, flush_interval=0.01)
    sink.start()

    await asyncio.to_thread(sink, _rec(node="from_thread"))   # 模拟 asyncio_0 线程
    await sink.stop()

    assert sink.dropped_no_loop == 0
    rows = await _rows()
    assert len(rows) == 1
    assert rows[0].node == "from_thread"


async def test_cross_thread_without_start_lands_via_bound_loop():
    """回归（同一个坑踩了第二次）：没人调 start() 时也必须能落库。

    上一次修复后我以为"回调改成 AsyncCallbackHandler 就没事了"，实际只是靠
    usage_log_sink.start() 提前绑定了 loop 才没露馅。走没有 lifespan 的运行方式
    （脚本、ASGI 直连）时，一篇论文 16 次调用 → 表里 0 行，stats 却显示"账平"。

    现在 loop 由 llm_scope 在事件循环线程里 capture_loop() 记住，
    sink 在回调线程里回退取用它，因此不依赖 start()。
    """
    sink = UsageLogSink(batch_size=10, flush_interval=0.01)
    assert sink._task is None, "用例前提：没有 start()"

    with llm_client.llm_scope(node="no_start", user_id=7):
        await asyncio.to_thread(sink, _rec(node="no_start"))   # 模拟 asyncio_0 线程

    await sink.stop()

    assert sink.dropped_no_loop == 0
    rows = await _rows()
    assert len(rows) == 1
    assert rows[0].node == "no_start"


async def test_flush_failure_counted_not_raised(monkeypatch):
    """写库失败不能让业务崩，也不能假装成功：必须计入 dropped_flush 并保持对账平衡。"""
    import app.core.llm_log_sink as mod

    class BoomSession:
        async def __aenter__(self):
            raise RuntimeError("database is locked")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(mod, "AsyncSessionLocal", lambda: BoomSession())

    sink = UsageLogSink(batch_size=10, flush_interval=0.01)
    sink(_rec())
    await sink.stop()          # 不应抛出

    stats = sink.stats()
    assert stats["flush_failures"] >= 1
    assert stats["written"] == 0
    assert stats["dropped_flush"] >= 1
    assert stats["balanced"] is True


async def test_accounting_balances_after_mixed_traffic():
    """混合场景下 收到 = 写入 + 各口径丢弃 + 未处理，这个等式必须成立。"""
    sink = UsageLogSink(batch_size=7, flush_interval=0.01)
    for i in range(33):
        sink(_rec(node=f"n{i % 3}"))
    await sink.stop()

    stats = sink.stats()
    assert stats["received"] == 33
    assert stats["written"] == 33
    assert stats["balanced"] is True


# ---------------------------------------------------------------------------
# 与计量层的串联
# ---------------------------------------------------------------------------
async def test_callback_record_lands_in_db():
    """端到端：模型回调产出的记录，经过 sink 后必须原样出现在表里。"""
    import uuid
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from app.core import llm_client

    sink = UsageLogSink(batch_size=10, flush_interval=0.01)
    original = list(llm_client._sinks)
    llm_client._sinks.clear()
    llm_client.register_sink(sink)
    try:
        msg = AIMessage(
            content="ok",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 800, "completion_tokens": 200,
                    "prompt_cache_hit_tokens": 300,
                },
                "model_name": "deepseek-flash",
            },
        )
        cb = llm_client.CostTrackingCallback()
        run_id = uuid.uuid4()
        sink.start()
        with llm_client.llm_scope(node="polish_paper", paper_id=11):
            # 走 async 回调路径（ainvoke 时的真实路径）；同步路径会在线程池里执行
            await cb.aon_chat_model_start({}, [], run_id=run_id)
            await cb.aon_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=run_id)
        await sink.stop()
    finally:
        llm_client._sinks.clear()
        llm_client._sinks.extend(original)

    rows = await _rows()
    assert len(rows) == 1
    assert rows[0].node == "polish_paper"
    assert rows[0].paper_id == 11
    assert rows[0].cache_hit_tokens == 300
    assert rows[0].total_tokens == 1000
    assert float(rows[0].cost) > 0

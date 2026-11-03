# core/llm_client.py —— LLM 调用治理层：计量 / 定价 / 超时 / 重试 / 降级
#
# 为什么要有这一层：模型调用散落在 6 个 Agent + 结构化封装 + 工具里，
# 每处各自 ainvoke，token 用量和成本根本无从统计（P0-2 之前连耗时都没记全）。
# 这里用两种手段收口，而不是要求每个调用点都改：
#   1) 构造收口：所有 ChatOpenAI 只从 build_chat_model() 出，超时/重试/回调一处配置
#   2) 计量收口：CostTrackingCallback 挂在模型实例上，横切所有 invoke/ainvoke，
#      包括 chain、with_structured_output、bind_tools 这些包装层
#
# 定价来源（2026-11-02 核对）：
#   - 官方定价页 https://api-docs.deepseek.com/quick_start/pricing
#   - 官方调价公告（2026-08-17 00:00 生效，峰谷双价）
# 两处口径存在差异（官方页为 USD、公告为 CNY），这里取人民币口径，
# 并在下方注释里留下对照，账单对不上时以官方控制台账单为准，可用 .env 覆盖倍率。
import asyncio
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger("app.llm.observability")

# 北京时间（UTC+8）。峰谷判定必须用北京时间：官方公告的"高峰"是北京时间
# 工作日 09:00-12:00、14:00-18:00，等价于 UTC 01:00-04:00、06:00-10:00。
BEIJING_TZ = timezone(timedelta(hours=8))

# ---------------------------------------------------------------------------
# 单价表：单位「元 / 百万 token」
#
# DeepSeek 计费有三个维度，缺一个成本就会算歪：
#   ① 输入分流：缓存命中 vs 未命中，两者差 30-60 倍
#   ② 输出单列：比输入贵 3 倍
#   ③ 峰谷双价：峰时是空闲时段的 2 倍
# 官方 USD 页对照（deepseek-flash，2026-09 核对）：
#   cache hit  off-peak $0.003 / peak $0.006
#   cache miss off-peak $0.15  / peak $0.30
#   output     off-peak $0.6   / peak $1.2
# ---------------------------------------------------------------------------
_PRICE_RMB_PER_1M: Dict[str, Dict[str, Dict[str, float]]] = {
    "deepseek-flash": {
        "peak": {"cache_hit_input": 0.10, "cache_miss_input": 3.00, "output": 9.00},
        "off_peak": {"cache_hit_input": 0.05, "cache_miss_input": 1.50, "output": 4.50},
    },
    "deepseek-v4-pro": {
        "peak": {"cache_hit_input": 0.30, "cache_miss_input": 9.00, "output": 27.00},
        "off_peak": {"cache_hit_input": 0.15, "cache_miss_input": 4.50, "output": 13.50},
    },
    # 演示模式的本地模型：不产生任何费用。
    # 补一条 0 价而不是让它落到「未知模型」分支：未知模型会打告警，
    # 而告警一旦成为常态，真正的缺价（影响成本故事）就会被淹掉。
    "demo-mock": {
        "peak": {"cache_hit_input": 0.0, "cache_miss_input": 0.0, "output": 0.0},
        "off_peak": {"cache_hit_input": 0.0, "cache_miss_input": 0.0, "output": 0.0},
    },
}

# 模型别名归一化。
# 关键事实（2026-11-02 实测）：配置写的是 deepseek-chat，但服务端响应里的
# model_name 是 "deepseek-flash" —— deepseek-chat/reasoner 已于 2026-07-24 下线，
# 仅作为别名被接受，实际由 V4.1-Flash 服务并按 Flash 计价。
# 所以计费一律按「响应里的模型名」归一化后再查表，不能按配置值查。
_MODEL_ALIASES: Dict[str, str] = {
    "deepseek-chat": "deepseek-flash",
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-v4-flash-vision-exp": "deepseek-flash",
    "deepseek-reasoner": "deepseek-v4-pro",
    "deepseek-v4-pro": "deepseek-v4-pro",
}

# 未知模型只告警一次，避免并发下把日志刷爆
_warned_models: set = set()


def normalize_model(model: Optional[str]) -> str:
    """把各种模型名收敛到单价表的 key。未知模型原样返回（后续按原价 0 处理并告警）。"""
    if not model:
        return "unknown"
    name = model.strip().lower()
    if name in _MODEL_ALIASES:
        return _MODEL_ALIASES[name]
    if name in _PRICE_RMB_PER_1M:
        return name
    # 前缀匹配：容忍 deepseek-flash-2026xxxx 这类带日期的快照名
    for known in _PRICE_RMB_PER_1M:
        if name.startswith(known):
            return known
    return name


def is_peak(ts: Optional[datetime] = None) -> bool:
    """判断某个时刻是否落在高峰计费时段（北京时间工作日 09-12、14-18）。

    已知简化：官方公告写明「中国法定节假日全天按空闲价」，这里没有节假日数据源，
    因此节假日会被判成峰时、成本估算偏高。宁可高估不低估 —— 低估会让人以为
    预算还有余量。
    """
    if not settings.LLM_PEAK_PRICING:
        return False
    now = (ts or datetime.now(BEIJING_TZ)).astimezone(BEIJING_TZ)
    if now.weekday() >= 5:          # 周六日全天空闲
        return False
    return (9 <= now.hour < 12) or (14 <= now.hour < 18)


def compute_cost(
    model: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    cache_hit_tokens: int = 0,
    *,
    peak: Optional[bool] = None,
) -> Decimal:
    """按 token 用量算钱（元）。返回 Decimal，避免浮点累加把成本算歪。"""
    key = normalize_model(model)
    table = _PRICE_RMB_PER_1M.get(key)
    if table is None:
        if key not in _warned_models:
            _warned_models.add(key)
            logger.warning("未知模型的单价缺失，成本按 0 计：model=%s（请在 llm_client 补单价表）", key)
        return Decimal("0")

    rate = table["peak" if (is_peak() if peak is None else peak) else "off_peak"]
    hit = max(0, min(cache_hit_tokens, prompt_tokens))     # 防御：命中数不可能超过输入数
    miss = max(0, prompt_tokens - hit)

    cost = (
        Decimal(hit) * Decimal(str(rate["cache_hit_input"]))
        + Decimal(miss) * Decimal(str(rate["cache_miss_input"]))
        + Decimal(max(0, completion_tokens)) * Decimal(str(rate["output"]))
    ) / Decimal(1_000_000)
    # 量化到 1e-8 元：单次调用常在 0.001 量级，位数不够会全变成 0
    return cost.quantize(Decimal("0.00000001"))


# ---------------------------------------------------------------------------
# 用量记录
# ---------------------------------------------------------------------------
@dataclass
class UsageRecord:
    """一次模型调用的用量与成本。字段与 llm_call_log 表一一对应。"""
    node: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cost: Decimal = Decimal("0")
    latency_ms: float = 0.0
    status: str = "ok"                  # ok / error
    paper_id: Optional[int] = None
    # 归属用户。为什么不靠 paper_id 反查：① 论文删除后流水要留存（无外键），
    # 反查就断了；② 选题推荐等调用根本没有 paper_id。缺了这列，按用户看成本
    # 会静默漏掉这两类记录 —— 那正是「控制成本」时最容易被忽略的账。
    user_id: Optional[int] = None
    peak: bool = False
    error: str = ""
    priced: bool = True                 # False = 单价缺失，cost 不可信
    extra: Dict[str, Any] = field(default_factory=dict)
    # 调用发生的准确时刻（UTC）。由记录时刻决定，而不是落库时刻 ——
    # 批量 flush 有 0.5-1s 延迟，跨天边界时用落库时间会把成本算到错误的日期上。
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_log(self) -> Dict[str, Any]:
        """日志/落库用的扁平日志字典（Decimal 转 float，便于 JSON 序列化）。"""
        return {
            "node": self.node,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "total_tokens": self.total_tokens,
            "cost": float(self.cost),
            "latency_ms": round(self.latency_ms, 1),
            "status": self.status,
            "paper_id": self.paper_id,
            "user_id": self.user_id,
            "peak": self.peak,
            "error": self.error[:300],
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# 消费端（sink）：第 1 步只写结构化日志，第 2 步由 DB sink 接手落表
# ---------------------------------------------------------------------------
_sinks: List[Callable[[UsageRecord], None]] = []


def register_sink(fn: Callable[[UsageRecord], None]) -> None:
    """注册一个消费端。要求：必须是非阻塞的（DB 落库内部自己排队）。"""
    if fn not in _sinks:
        _sinks.append(fn)


def _default_log_sink(rec: UsageRecord) -> None:
    import json
    logger.info("LLM_CALL %s", json.dumps(rec.as_log(), ensure_ascii=False))


register_sink(_default_log_sink)


def emit(rec: UsageRecord) -> None:
    """把用量记录广播给所有 sink。sink 内部异常不允许影响业务调用。"""
    for fn in list(_sinks):
        try:
            fn(rec)
        except Exception:  # noqa: BLE001 —— 观测失败绝不能拖垮主流程
            logger.exception("用量 sink 执行失败: %s", getattr(fn, "__name__", fn))


# ---------------------------------------------------------------------------
# 事件循环引用
#
# 实测（langchain-core 0.3.63）：即使回调继承 AsyncCallbackHandler、实现了 aon_llm_end，
# ainvoke 下真正被调用的仍是**同步** on_llm_end，且在 executor 线程（asyncio_0）里执行。
# 那个线程里 asyncio.get_running_loop() 直接抛 RuntimeError，所以异步落库在这里
# 拿不到 loop。解决方式不是"记得在启动时调 start()"，而是在事件循环线程里
# 顺手把 loop 记下来：llm_scope 一定在主线程/事件循环里进入。
# ---------------------------------------------------------------------------
_loop_ref: Optional[asyncio.AbstractEventLoop] = None


def capture_loop() -> Optional[asyncio.AbstractEventLoop]:
    """在事件循环线程里调用：记住当前 loop，供回调（可能在别的线程）投递用。

    已关闭的 loop 会被替换 —— 脚本/测试每轮新建 loop，不刷新就会一律投递失败。
    """
    global _loop_ref
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return _loop_ref
    if _loop_ref is None or _loop_ref.is_closed() or _loop_ref is not loop:
        _loop_ref = loop
    return _loop_ref


def get_bound_loop() -> Optional[asyncio.AbstractEventLoop]:
    return _loop_ref


# ---------------------------------------------------------------------------
# 上下文：callback 只能拿到"这次算什么"，拿不到"算在哪个节点/哪篇论文"
# 用 contextvar 传递，避免一路给每个函数加参数
# ---------------------------------------------------------------------------
_ctx: ContextVar[Dict[str, Any]] = ContextVar("llm_call_ctx", default={})


@contextmanager
def llm_scope(
    node: Optional[str] = None,
    paper_id: Optional[int] = None,
    user_id: Optional[int] = None,
    **extra,
):
    """给一段代码里的所有模型调用打上 node / paper_id / user_id 标签。

    并发安全：contextvar 在 asyncio.create_task 时会被复制到子任务，
    因此每章写作并发时各自 scope 互不串台。
    """
    # 在事件循环线程里顺手记下 loop：回调是同步的、跑在线程池里，
    # 那边拿不到 loop，只能靠这里提前记住（详见 _loop_ref 注释）。
    capture_loop()
    current = dict(_ctx.get())
    if node is not None:
        current["node"] = node
    if paper_id is not None:
        current["paper_id"] = paper_id
    if user_id is not None:
        current["user_id"] = user_id
    current.update(extra)
    token = _ctx.set(current)
    try:
        yield
    finally:
        _ctx.reset(token)


def _ctx_node() -> str:
    return _ctx.get().get("node") or "unknown"


def _ctx_paper_id() -> Optional[int]:
    return _ctx.get().get("paper_id")


def _ctx_user_id() -> Optional[int]:
    return _ctx.get().get("user_id")


# ---------------------------------------------------------------------------
# 用量提取：不同层级的对象结构不一样，逐个兼容
# ---------------------------------------------------------------------------
def extract_usage(message: Any) -> tuple:
    """从 AIMessage（或 LLMResult 里的 message）提取 (prompt, completion, cache_hit, model)。

    三个来源按可靠性排序：
      1) response_metadata.token_usage —— DeepSeek 原生字段，含 prompt_cache_hit_tokens
      2) usage_metadata —— LangChain 标准字段，input_token_details.cache_read
      3) 都没有 → (0, 0, 0)，说明该调用无法计量（要如实暴露，不能假装成功）
    """
    if message is None:
        return 0, 0, 0, None

    prompt = completion = cache_hit = 0
    model = None

    meta = getattr(message, "response_metadata", None) or {}
    usage = meta.get("token_usage") or {}
    if usage:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        # DeepSeek 直接给了缓存命中数，这是最准的口径
        cache_hit = int(usage.get("prompt_cache_hit_tokens") or 0)
        if not cache_hit:
            details = usage.get("prompt_tokens_details") or {}
            cache_hit = int(details.get("cached_tokens") or 0)
    model = meta.get("model_name") or model

    um = getattr(message, "usage_metadata", None) or {}
    if um:
        prompt = prompt or int(um.get("input_tokens") or 0)
        completion = completion or int(um.get("output_tokens") or 0)
        cache_read = (um.get("input_token_details") or {}).get("cache_read") or 0
        cache_hit = max(cache_hit, int(cache_read or 0))

    return prompt, completion, cache_hit, model


# ---------------------------------------------------------------------------
# 回调：横切所有模型调用
# ---------------------------------------------------------------------------
class CostTrackingCallback(AsyncCallbackHandler):
    """挂在 ChatOpenAI 上，每次模型调用结束就产出一条 UsageRecord。

    为什么用回调而不是改调用点：项目里有 6 个 Agent、chain 组合、with_structured_output
    的二次包装、bind_tools 的评审图。逐个改一定漏；回调挂在模型实例上，
    这些包装层最终都要经过模型 invoke，天然全覆盖。

    为什么继承 AsyncCallbackHandler：理论上异步运行会走 aon_* 方法。但实测
    （langchain-core 0.3.63 + langchain-openai）**并非如此**：ainvoke 下真正被调用的
    仍是同步 on_llm_end，而且在线程池（线程名 asyncio_0）里执行，那里
    asyncio.get_running_loop() 会直接抛错。这个坑踩过两次：
      - 第一次：只实现同步方法 → 整批记录按"没有事件循环"被丢弃（表里 0 行，
        日志却一切正常）；
      - 第二次：以为改成 AsyncCallbackHandler 就解决了，实际上只是靠
        usage_log_sink.start() 事先绑定了 loop 才没露馅；一旦没人调 start()
        （脚本 / 无 lifespan 的运行方式），照样整批丢。
    现在的做法：同步方法照常记录（本环境实际走的就是它），事件循环由
    llm_scope 里的 capture_loop() 记住，sink 用跨线程投递兜底。
    aon_* 版本保留 —— 未来 langchain 修正了行为，两边都能覆盖。
    """

    # 同一实例被并发使用（章节并发 4）时，用 run_id 区分各自的开始时间
    def __init__(self) -> None:
        self._started: Dict[Any, float] = {}

    # --- 计时 ---
    def _mark_start(self, run_id) -> None:
        self._started[run_id] = time.perf_counter()

    def _latency(self, run_id) -> float:
        t0 = self._started.pop(run_id, None)
        return (time.perf_counter() - t0) * 1000 if t0 else 0.0

    # --- 取 response 里的 message ---
    @staticmethod
    def _first_message(response: Any) -> Any:
        try:
            generations = getattr(response, "generations", None) or []
            if generations and generations[0]:
                return getattr(generations[0][0], "message", None)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _record_ok(self, run_id, response, latency_ms: Optional[float] = None) -> None:
        message = self._first_message(response)
        prompt, completion, cache_hit, model = extract_usage(message)
        if latency_ms is None:
            latency_ms = self._latency(run_id)

        # 模型名缺失时回退到配置值，并标注 priced 可信度
        model = model or settings.DEEPSEEK_MODEL
        key = normalize_model(model)
        priced = key in _PRICE_RMB_PER_1M
        peak = is_peak()

        emit(UsageRecord(
            node=_ctx_node(),
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cache_hit_tokens=cache_hit,
            cost=compute_cost(model, prompt, completion, cache_hit, peak=peak),
            latency_ms=latency_ms,
            status="ok",
            paper_id=_ctx_paper_id(),
            user_id=_ctx_user_id(),
            peak=peak,
            priced=priced,
            extra={} if (prompt or completion) else {"unmetered": True},
        ))

    def _record_error(self, run_id, error: BaseException) -> None:
        emit(UsageRecord(
            node=_ctx_node(),
            model=settings.DEEPSEEK_MODEL,
            latency_ms=self._latency(run_id),
            status="error",
            paper_id=_ctx_paper_id(),
            user_id=_ctx_user_id(),
            error=f"{type(error).__name__}: {error}",
            priced=False,
        ))

    # 同步 / 异步两条路径都要实现：langchain 在 ainvoke 时对 AsyncCallbackHandler
    # 调 a* 方法（在事件循环线程），对同步 handler 则丢到线程池；只实现一边会漏计量。
    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs) -> None:
        self._mark_start(run_id)

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs) -> None:
        self._mark_start(run_id)

    def on_llm_end(self, response, *, run_id=None, **kwargs) -> None:
        self._record_ok(run_id, response)

    def on_llm_error(self, error, *, run_id=None, **kwargs) -> None:
        self._record_error(run_id, error)

    async def aon_llm_start(self, serialized, prompts, *, run_id=None, **kwargs) -> None:
        self._mark_start(run_id)

    async def aon_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs) -> None:
        self._mark_start(run_id)

    async def aon_llm_end(self, response, *, run_id=None, **kwargs) -> None:
        self._record_ok(run_id, response)

    async def aon_llm_error(self, error, *, run_id=None, **kwargs) -> None:
        self._record_error(run_id, error)


# ---------------------------------------------------------------------------
# 模型构造收口
# ---------------------------------------------------------------------------
def build_chat_model(
    *,
    model: Optional[str] = None,
    temperature: float = 0.7,
    streaming: bool = False,
    max_tokens: Optional[int] = None,
    callbacks: Optional[List[Any]] = None,
) -> Any:
    """唯一的 ChatOpenAI 构造入口：超时 / 重试 / 计量回调在这里统一配置。

    重试说明：这里不再自己写退避循环。langchain-openai 的 max_retries 透传给
    OpenAI SDK，SDK 内部就是指数退避 + 抖动。若在外层再包 3 次，实际会放大成
    3×3=9 次请求 —— 限流时反而把配额打得更狠。

    演示模式（DEMO_MODE）：返回 DemoChatModel —— 它同样是 BaseChatModel，
    因此 bind_tools / with_structured_output / 流式与计量回调的行为完全一致，
    上层 Agent 不需要任何改动。计量照常记录（token 数照估，单价查不到 → 成本记 ¥0）。
    """
    limit = max_tokens or settings.LLM_MAX_TOKENS

    if settings.DEMO_MODE:
        from app.demo.chat import DemoChatModel
        return DemoChatModel(
            temperature=temperature,
            max_tokens=limit,
            streaming=streaming,
            callbacks=(callbacks if callbacks is not None else [CostTrackingCallback()]),
        )

    return ChatOpenAI(
        model=model or settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=temperature,
        streaming=streaming,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.LLM_MAX_RETRIES,
        # 关键：**不要**传 max_tokens= 参数。
        # langchain-openai 0.2.x 会把这个值改写成 max_completion_tokens 再发给服务端，
        # 而 DeepSeek 的兼容接口不认这个字段 —— 实测（2026-11-02）：
        #   请求带 max_tokens=16           → finish_reason=length，只出 47 字符（生效）
        #   请求带 max_completion_tokens=16 → finish_reason=stop，照常输出 246 tokens（被忽略）
        # extra_body 是 OpenAI SDK 的透传口，能原样带上 max_tokens，实测生效。
        extra_body={"max_tokens": limit},
        callbacks=(callbacks if callbacks is not None else [CostTrackingCallback()]),
    )


def build_embedding_model():
    """Embedding 模型的唯一构造入口（计量暂不覆盖：百炼按 token 计价口径不同）。

    演示模式返回本地哈希向量：不联网，但依然按字符重合度排序，
    检索结果与查询主题相关（不是随机返回），RAG 链路可被真实演示。
    """
    if settings.DEMO_MODE:
        from app.demo.chat import DemoHashEmbeddings
        return DemoHashEmbeddings()

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=settings.DASHSCOPE_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        check_embedding_ctx_length=False,
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )

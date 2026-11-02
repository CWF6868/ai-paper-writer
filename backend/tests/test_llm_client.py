# test_llm_client.py —— LLM 调用治理层：计量、定价、峰谷、标签隔离
#
# 这一层的 bug 不会让程序崩，只会让成本表算错 —— 所以测试重点不是"能跑"，
# 而是几个真会算错钱的地方：峰谷判定、缓存命中抵扣、模型别名、并发标签串台。
import asyncio
import uuid
from datetime import datetime
from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.config import settings
from app.core import llm_client
from app.core.llm_client import (
    CostTrackingCallback,
    UsageRecord,
    compute_cost,
    extract_usage,
    is_peak,
    llm_scope,
    normalize_model,
)


@pytest.fixture
def captured():
    """临时接管 sink 列表，用例结束后恢复，避免污染其他测试。"""
    original = list(llm_client._sinks)
    box = []
    llm_client._sinks.clear()
    llm_client.register_sink(box.append)
    yield box
    llm_client._sinks.clear()
    llm_client._sinks.extend(original)


# ---------------------------------------------------------------------------
# 模型名归一化
# ---------------------------------------------------------------------------
def test_alias_deepseek_chat_maps_to_flash():
    """配置里写的是 deepseek-chat，实际由 deepseek-flash 服务并按 Flash 计价。

    如果不归一化，成本表会查不到价 → 全部记 0，成本故事直接失效。
    """
    assert normalize_model("deepseek-chat") == "deepseek-flash"
    assert normalize_model("DeepSeek-Chat") == "deepseek-flash"       # 大小写不敏感
    assert normalize_model("deepseek-v4-flash") == "deepseek-flash"   # 旧名仍被接受


def test_alias_snapshot_name_prefix_match():
    """带日期快照的模型名（deepseek-flash-2026xxxx）也要能匹配到价目。"""
    assert normalize_model("deepseek-flash-2026-08-17") == "deepseek-flash"


def test_alias_unknown_model_passthrough():
    """未知模型原样返回，不当成 flash 计价 —— 宁可标缺失，不能张冠李戴。"""
    assert normalize_model("gpt-5.4") == "gpt-5.4"
    assert normalize_model(None) == "unknown"


# ---------------------------------------------------------------------------
# 峰谷判定
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dt, expected, why", [
    (datetime(2026, 11, 2, 10, 0), True, "周一 10:00 峰时"),
    (datetime(2026, 11, 2, 12, 30), False, "午休 12:30 谷时"),
    (datetime(2026, 11, 2, 15, 0), True, "周一 15:00 峰时"),
    (datetime(2026, 11, 2, 19, 0), False, "晚间 19:00 谷时"),
    (datetime(2026, 11, 2, 8, 59), False, "08:59 尚未进峰"),
    (datetime(2026, 11, 7, 10, 0), False, "周六全天空闲"),
])
def test_is_peak_boundaries(dt, expected, why, monkeypatch):
    """峰谷判定按北京时间，边界值最容易写错（9/12/14/18 的闭开区间）。"""
    monkeypatch.setattr(settings, "LLM_PEAK_PRICING", True)
    assert is_peak(dt.replace(tzinfo=llm_client.BEIJING_TZ)) is expected, why


def test_peak_switch_off(monkeypatch):
    """关掉峰谷开关后一律按空闲价 —— 这是给 .env 留的逃生口。"""
    monkeypatch.setattr(settings, "LLM_PEAK_PRICING", False)
    assert is_peak(datetime(2026, 11, 2, 10, 0).replace(tzinfo=llm_client.BEIJING_TZ)) is False


# ---------------------------------------------------------------------------
# 算钱
# ---------------------------------------------------------------------------
def test_cost_cache_hit_is_much_cheaper():
    """同样 1000 输入 token，缓存命中与未命中差 60 倍。

    这是成本故事的核心：命中率从 0 提到 50%，输入侧成本直接砍一半。
    """
    miss = compute_cost("deepseek-flash", 1000, 0, 0, peak=False)
    hit = compute_cost("deepseek-flash", 1000, 0, 1000, peak=False)
    assert miss == Decimal("0.0015")        # 1000 * 1.5 / 1e6
    assert hit == Decimal("0.00005")        # 1000 * 0.05 / 1e6
    assert miss / hit == Decimal("30")


def test_cost_peak_is_double_off_peak():
    """峰时价是空闲价的 2 倍，差一个 is_peak 判断就少算一半成本。"""
    off = compute_cost("deepseek-flash", 0, 1_000_000, 0, peak=False)
    on = compute_cost("deepseek-flash", 0, 1_000_000, 0, peak=True)
    assert off == Decimal("4.5")
    assert on == Decimal("9.0")


def test_cost_mixed_hit_and_miss():
    """命中与未命中混合时，两部分要分别计价，不能整单按一个价算。"""
    # 1000 输入中 400 命中：400*0.05 + 600*1.5 = 20 + 900 = 920 → /1e6
    got = compute_cost("deepseek-flash", 1000, 0, 400, peak=False)
    assert got == Decimal("0.00092")


def test_cost_cache_hit_clamped_to_prompt():
    """命中数超过输入数是脏数据（服务端口径变化时会出现），钳住而不是算出负成本。"""
    got = compute_cost("deepseek-flash", 100, 0, 999, peak=False)
    assert got == Decimal("0.000005")       # 全部按命中算，不会出现负的未命中


def test_cost_unknown_model_is_zero_and_warns_once(captured):
    """未知模型成本记 0，但必须告警（否则成本表悄悄少钱没人发现）。"""
    assert compute_cost("some-new-model", 1000, 1000) == Decimal("0")
    assert compute_cost("some-new-model", 1000, 1000) == Decimal("0")   # 第二次不再告警


# ---------------------------------------------------------------------------
# 用量提取
# ---------------------------------------------------------------------------
def test_extract_usage_prefers_deepseek_native_fields():
    """DeepSeek 的 token_usage 里有 prompt_cache_hit_tokens，这是最准的命中口径。"""
    msg = AIMessage(
        content="x",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 369,
                "completion_tokens": 12,
                "prompt_cache_hit_tokens": 128,
            },
            "model_name": "deepseek-flash",
        },
    )
    assert extract_usage(msg) == (369, 12, 128, "deepseek-flash")


def test_extract_usage_falls_back_to_usage_metadata():
    """部分路径只有 usage_metadata，此时取 input_token_details.cache_read。"""
    msg = AIMessage(
        content="x",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_token_details": {"cache_read": 64},
        },
    )
    assert extract_usage(msg) == (100, 20, 64, None)


def test_extract_usage_missing_returns_zeros():
    """拿不到用量时返回 0，而不是抛异常 —— 计量失败不该中断生成。"""
    msg = AIMessage(content="x")
    assert extract_usage(msg) == (0, 0, 0, None)
    assert extract_usage(None) == (0, 0, 0, None)


# ---------------------------------------------------------------------------
# 回调
# ---------------------------------------------------------------------------
def _fake_llm_result(prompt=100, completion=50, cache_hit=0, model="deepseek-flash"):
    msg = AIMessage(
        content="ok",
        response_metadata={
            "token_usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "prompt_cache_hit_tokens": cache_hit,
            },
            "model_name": model,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def test_callback_emits_record_with_node_and_paper(captured):
    """回调产出的记录要带上下文标签，否则成本无法归因到节点/论文。"""
    cb = CostTrackingCallback()
    run_id = uuid.uuid4()
    with llm_scope(node="write_sections", paper_id=42):
        cb.on_chat_model_start({}, [], run_id=run_id)
        cb.on_llm_end(_fake_llm_result(prompt=1000, completion=500, cache_hit=200), run_id=run_id)

    assert len(captured) == 1
    rec = captured[0]
    assert rec.node == "write_sections"
    assert rec.paper_id == 42
    assert (rec.prompt_tokens, rec.completion_tokens, rec.cache_hit_tokens) == (1000, 500, 200)
    assert rec.status == "ok"
    assert rec.cost > 0


def test_callback_latency_measured(captured):
    """延迟必须落在合理区间，不能恒为 0（否则延迟聚合全是 0，看不出慢在哪）。"""
    cb = CostTrackingCallback()
    run_id = uuid.uuid4()
    cb.on_chat_model_start({}, [], run_id=run_id)
    cb.on_llm_end(_fake_llm_result(), run_id=run_id)
    assert 0 <= captured[0].latency_ms < 10_000


def test_callback_error_recorded(captured):
    """失败也要落一条 status=error，否则失败调用在成本表里完全消失。"""
    cb = CostTrackingCallback()
    run_id = uuid.uuid4()
    cb.on_chat_model_start({}, [], run_id=run_id)
    cb.on_llm_error(TimeoutError("read timeout"), run_id=run_id)

    assert len(captured) == 1
    assert captured[0].status == "error"
    assert "TimeoutError" in captured[0].error
    assert captured[0].priced is False


def test_callback_unmetered_flagged(captured):
    """拿不到 token 的成功调用要打 unmetered 标记，不能静默当成 0 消耗。"""
    cb = CostTrackingCallback()
    run_id = uuid.uuid4()
    cb.on_chat_model_start({}, [], run_id=run_id)
    cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]]), run_id=run_id)

    assert captured[0].extra.get("unmetered") is True
    assert captured[0].total_tokens == 0


def test_sink_failure_does_not_break_call(captured):
    """观测层出事绝不能拖垮业务：一个 sink 抛异常，其他 sink 仍要收到。"""
    received = []

    def boom(_rec):
        raise RuntimeError("db down")

    llm_client.register_sink(boom)
    llm_client.register_sink(received.append)

    cb = CostTrackingCallback()
    run_id = uuid.uuid4()
    cb.on_chat_model_start({}, [], run_id=run_id)
    cb.on_llm_end(_fake_llm_result(), run_id=run_id)   # 不应抛出

    assert len(received) == 1


# ---------------------------------------------------------------------------
# 并发：标签隔离（P0-2 的章节并发会同时跑 4 个写作调用）
# ---------------------------------------------------------------------------
async def test_scope_isolated_under_concurrency(captured):
    """并发下每个协程的 node 标签必须各自独立，否则成本会记到错误的节点上。"""
    cb = CostTrackingCallback()

    async def worker(node, delay):
        with llm_scope(node=node, paper_id=node):
            await asyncio.sleep(delay)
            run_id = uuid.uuid4()
            cb.on_chat_model_start({}, [], run_id=run_id)
            cb.on_llm_end(_fake_llm_result(prompt=10, completion=1), run_id=run_id)

    await asyncio.gather(
        worker("node_slow", 0.03),
        worker("node_fast", 0.0),
        worker("node_mid", 0.01),
    )

    pairs = sorted((r.node, r.paper_id) for r in captured)
    assert pairs == [("node_fast", "node_fast"), ("node_mid", "node_mid"), ("node_slow", "node_slow")]


async def test_scope_restores_after_exit(captured):
    """scope 退出后必须恢复上下文，否则标签会泄漏到后续无关调用。"""
    with llm_scope(node="inner"):
        pass
    cb = CostTrackingCallback()
    run_id = uuid.uuid4()
    cb.on_chat_model_start({}, [], run_id=run_id)
    cb.on_llm_end(_fake_llm_result(), run_id=run_id)
    assert captured[0].node == "unknown"


# ---------------------------------------------------------------------------
# 构造收口
# ---------------------------------------------------------------------------
def test_build_chat_model_has_timeout_and_callback():
    """超时和计量回调必须挂在模型实例上 —— 这是全项目唯一的构造出口。"""
    from app.utils.llm import get_llm

    llm = get_llm()
    assert any(isinstance(c, CostTrackingCallback) for c in llm.callbacks)
    assert settings.LLM_TIMEOUT_SECONDS == 60.0
    # ChatOpenAI 把 timeout 存在 request_timeout / client.timeout 里，兼容两种
    timeout = getattr(llm, "request_timeout", None)
    if timeout is None:
        timeout = getattr(getattr(llm, "client", None), "timeout", None)
    assert timeout is not None


def test_usage_record_log_shape():
    """日志/落库的字段形状要稳定，第 2 步的建表直接照它来。"""
    rec = UsageRecord(
        node="write_sections", model="deepseek-flash",
        prompt_tokens=1000, completion_tokens=500, cache_hit_tokens=200,
        cost=Decimal("0.00123"), latency_ms=1234.56, paper_id=7,
    )
    log = rec.as_log()
    assert log["total_tokens"] == 1500
    assert log["cost"] == pytest.approx(0.00123)
    assert log["latency_ms"] == 1234.6
    assert set(log) >= {
        "node", "model", "prompt_tokens", "completion_tokens", "cache_hit_tokens",
        "total_tokens", "cost", "latency_ms", "status", "paper_id", "peak", "error",
    }

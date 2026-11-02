# 单元测试：P0-1 结构化输出 + 工具调用（全部离线，不联网、不花 token）
#
# 覆盖设计文档 5.2 节要求的 6 项：
#   test_review_schema_valid        维度缺失/重复/越界分数 → ValidationError
#   test_review_avg_reconcile       综合分与均值偏差 >0.03 时被纠正
#   test_structured_fallback_chain  三级降级逐级记录，最终 raw_parse 成功
#   test_structured_all_fail        三级全失败 → StructuredOutputError
#   test_citation_tool_gray_zone    灰区断言调了 LLM；非灰区断言没调
#   test_tool_budget                超过 MAX_TOOL_ROUNDS / MAX_TOOL_CALLS 强制 finalize
# 另附 search_literature 的编号契约测试。
import asyncio
import logging

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool as lc_tool
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

import app.llm.tools.citation as citation
from app.agents.reviewer_graph import run_reviewer
from app.llm.schemas import (
    CitationVerdict,
    Dimension,
    DimensionScore,
    ReviewResult,
)
from app.llm.structured import call_structured, StructuredOutputError
from app.llm.tools.citation import verify_citation
from app.llm.tools.context import set_current_references
from app.llm.tools.literature import search_literature

# 便捷的合法评审结果 JSON（供 finalize 收口 / 灰区裁决使用）
VALID_REVIEW_JSON = (
    '{"score": 0.82, "dimensions": ['
    '{"dimension": "创新性与学术价值", "score": 0.8, "comment": "有一定创新"},'
    '{"dimension": "逻辑结构与论证", "score": 0.85, "comment": "论证清晰"},'
    '{"dimension": "内容完整度与充分性", "score": 0.8, "comment": "内容完整"},'
    '{"dimension": "语言表达与学术规范", "score": 0.83, "comment": "语言规范"},'
    '{"dimension": "文献引用质量", "score": 0.8, "comment": "引用恰当"}'
    '], "overall": "总体不错，建议补充对比实验。"}'
)


def _dims(score: float = 0.8) -> list:
    """构造齐整的 5 维打分"""
    return [DimensionScore(dimension=d, score=score, comment="一句话点评") for d in Dimension]


class _Out(BaseModel):
    """最小结构化输出 Schema（测降级链用）"""
    score: float = Field(description="分数")


class ScriptedChatModel(BaseChatModel):
    """按剧本吐消息的假模型：支持 bind_tools（让 function_calling 分支真实可走通）。"""

    script: list
    _i: int = PrivateAttr(default=0)

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self):
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        idx = min(self._i, len(self.script) - 1)
        msg = self.script[idx]
        self._i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


# ==================== 1) Schema 校验 ====================

def test_review_schema_valid():
    # 维度缺失（4 项）→ 拒绝
    with pytest.raises(ValidationError):
        ReviewResult(score=0.8, dimensions=_dims()[:4], overall="x")
    # 维度重复 → 拒绝
    dup = [DimensionScore(dimension=Dimension.INNOVATION, score=0.8, comment="c")] * 5
    with pytest.raises(ValidationError):
        ReviewResult(score=0.8, dimensions=dup, overall="x")
    # 综合分越界 → 拒绝（不再靠手写 clamp）
    with pytest.raises(ValidationError):
        ReviewResult(score=1.7, dimensions=_dims(), overall="x")
    # 维度分越界 → 拒绝
    with pytest.raises(ValidationError):
        DimensionScore(dimension=Dimension.INNOVATION, score=1.2, comment="c")
    # 未知维度名（自由字符串）→ 拒绝
    with pytest.raises(ValidationError):
        ReviewResult(
            score=0.8,
            dimensions=[
                DimensionScore(dimension=d, score=0.8, comment="c")
                for d in ["创新性与学术价值", "逻辑结构与论证", "内容完整度与充分性",
                          "语言表达与学术规范", "自定义维度"]
            ],
            overall="x",
        )
    # 合法输入 → 通过
    assert ReviewResult(score=0.8, dimensions=_dims(), overall="x").score == 0.8


def test_review_avg_reconcile():
    # 偏差 > 0.03 → 以维度均值静默纠正
    r = ReviewResult(score=0.9, dimensions=_dims(0.6), overall="x")
    assert r.score == 0.6
    # 偏差 ≤ 0.03 → 保留模型给的综合分
    r2 = ReviewResult(score=0.82, dimensions=_dims(0.8), overall="x")
    assert abs(r2.score - 0.82) < 1e-9


def test_review_tolerates_ten_point_scale():
    """模型按 10 分制打分（2026-11-02 评测实测：score=2.2 直接导致整篇降级）。
    整体换量纲时归一化救回，而不是让评审作废。"""
    names = [d.value for d in Dimension]
    payload = {
        "score": 2.2,
        "overall": "整体存在明显缺陷",
        "dimensions": [
            {"dimension": n, "score": s, "comment": "c"}
            for n, s in zip(names, [2.2, 2.4, 2.0, 2.5, 2.1])
        ],
    }
    r = ReviewResult.model_validate(payload)
    assert abs(r.score - 0.22) < 1e-6
    assert all(0.0 <= d.score <= 1.0 for d in r.dimensions)


def test_review_rejects_isolated_out_of_range():
    """只有个别字段越界 = 垃圾输出，不能靠除以 10 洗成合法值。"""
    with pytest.raises(ValidationError):
        ReviewResult.model_validate({
            "score": 1.7,
            "overall": "x",
            "dimensions": [
                {"dimension": d.value, "score": 0.8, "comment": "c"} for d in Dimension
            ],
        })


def test_review_unwraps_nested_and_aliases():
    """外层包装 {"review": {...}} + 别名 overall_score/summary（json_mode 实测形状）"""
    names = [d.value for d in Dimension]
    payload = {
        "review": {
            "overall_score": 6.5,
            "summary": "本文围绕……",
            "dimensions": {
                n: {"score": s, "comment": "具体缺陷"}
                for n, s in zip(names, [6.5, 7.0, 6.0, 7.0, 6.5])
            },
            "section_feedback": [],
            "must_fix": [],
        }
    }
    r = ReviewResult.model_validate(payload)
    assert abs(r.score - 0.65) < 1e-6        # 别名 + 10 分制都已归一
    assert r.overall == "本文围绕……"         # summary → overall
    assert len(r.dimensions) == 5


# ==================== 2) 三级降级链 ====================

async def test_structured_fallback_chain(caplog):
    """Fake 模型不支持 tools → function_calling/json_mode 逐级失败并记录 → raw_parse 成功"""
    model = FakeListChatModel(responses=['{"score": 0.5}'])
    with caplog.at_level(logging.INFO, logger="app.llm.observability"):
        parsed, mode = await call_structured(model, _Out, [("human", "x")], node="test_chain")
    assert mode == "raw_parse"
    assert parsed.score == 0.5
    msgs = [r.getMessage() for r in caplog.records if r.name == "app.llm.observability"]
    assert any('"mode": "function_calling"' in m and '"status": "fail"' in m for m in msgs)
    assert any('"mode": "json_mode"' in m and '"status": "fail"' in m for m in msgs)
    assert any('"mode": "raw_parse"' in m and '"status": "ok"' in m for m in msgs)


async def test_structured_function_calling_first_choice():
    """支持 tools 的模型 → 第一级 function_calling 直接成功，无降级"""
    tool_msg = AIMessage(content="", tool_calls=[
        {"name": "_Out", "args": {"score": 0.5}, "id": "c1", "type": "tool_call"},
    ])
    model = ScriptedChatModel(script=[tool_msg])
    parsed, mode = await call_structured(model, _Out, [("human", "x")], node="test_fc")
    assert mode == "function_calling"
    assert parsed.score == 0.5


async def test_structured_all_fail():
    """三级全失败 → 显式抛 StructuredOutputError（绝不静默返回脏数据）"""
    model = FakeListChatModel(responses=["这不是 JSON"])
    with pytest.raises(StructuredOutputError):
        await call_structured(model, _Out, [("human", "x")], node="test_fail")


# ==================== 3) 引文查证工具：灰区两级判定 ====================

REF_TEXT = "检索增强生成技术通过外部知识库提升大模型回答的准确性"
REFS = [{"title": "文献A", "text": REF_TEXT}]

# 假向量：与 REF_TEXT 的标准向量 [1, 0] 配出指定余弦值
CLAIM_SUPPORTED = "检索增强生成技术可以提升大模型回答的准确性"      # cos≈0.99
CLAIM_UNSUPPORTED = "今天股市行情非常好"                            # cos=0
CLAIM_GRAY = "知识库可以缓解模型幻觉"                              # cos=0.5 → 灰区
VECTORS = {
    REF_TEXT: [1.0, 0.0],
    CLAIM_SUPPORTED: [0.99, 0.141],
    CLAIM_UNSUPPORTED: [0.0, 1.0],
    CLAIM_GRAY: [0.6, 0.8],
}


async def test_citation_tool_gray_zone(monkeypatch):
    """非灰区纯计算不调 LLM；只有 [0.35, 0.72] 灰区才升级裁决（控成本的关键）"""
    monkeypatch.setattr(citation, "_embed", lambda text: VECTORS[text])

    judge_calls = []

    async def fake_judge(claim, ref_text, sim):
        judge_calls.append((claim, round(sim, 2)))
        return CitationVerdict(verdict="partial", similarity=sim, reason="灰区裁决")

    monkeypatch.setattr(citation, "_llm_judge", fake_judge)

    with set_current_references(REFS):
        # 高相似 + 关键词重合 → supported，纯计算
        out = await verify_citation.ainvoke({"ref_index": 1, "claim": CLAIM_SUPPORTED})
        assert out["verdict"] == "supported"
        # 低相似 → unsupported，纯计算
        out = await verify_citation.ainvoke({"ref_index": 1, "claim": CLAIM_UNSUPPORTED})
        assert out["verdict"] == "unsupported"
        assert judge_calls == []          # 非灰区：一次 LLM 都没调

        # 灰区 → 升级 LLM 裁决
        out = await verify_citation.ainvoke({"ref_index": 1, "claim": CLAIM_GRAY})
        assert out["verdict"] == "partial"
        assert judge_calls == [(CLAIM_GRAY, 0.6)]

        # 编号越界 → not_found
        out = await verify_citation.ainvoke({"ref_index": 99, "claim": CLAIM_SUPPORTED})
        assert out["verdict"] == "not_found"


# ==================== 4) 工具预算控制 ====================

COUNTER = {"n": 0}


@lc_tool
def echo_probe(x: int) -> str:
    """探测工具：数自己被实际执行了几次"""
    COUNTER["n"] += 1
    return f"pong-{x}"


def _tc(i: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": "echo_probe", "args": {"x": i}, "id": f"call_{i}", "type": "tool_call"},
    ])


async def test_tool_budget_rounds():
    """模型无限想调工具 → MAX_TOOL_ROUNDS=2 后强制 finalize（成本控制写成测试）"""
    COUNTER["n"] = 0
    valid = AIMessage(content=VALID_REVIEW_JSON)
    # 剧本：3 条 tool_call 消息（永远调不完）+ 兜底 JSON（供 finalize 收口）
    model = ScriptedChatModel(script=[_tc(1), _tc(2), _tc(3), valid, valid, valid])
    result, mode = await run_reviewer(model, [("human", "请评审")], tools=[echo_probe])

    assert COUNTER["n"] == 2                       # 只执行了 2 轮，第 3 次被强制收口
    assert result.score == 0.82
    assert mode == "raw_parse"                     # Scripted 模型上结构化收口走裸解析


async def test_tool_budget_total_calls():
    """累计调用数超过 MAX_TOOL_CALLS=6 → 即使轮数没用完也强制 finalize"""
    COUNTER["n"] = 0
    valid = AIMessage(content=VALID_REVIEW_JSON)

    def _tc3(i: int) -> AIMessage:
        return AIMessage(content="", tool_calls=[
            {"name": "echo_probe", "args": {"x": j}, "id": f"call_{i}_{j}", "type": "tool_call"}
            for j in range(3)
        ])

    model = ScriptedChatModel(script=[_tc3(1), _tc3(2), valid, valid, valid])
    result, _ = await run_reviewer(model, [("human", "请评审")], tools=[echo_probe])

    assert COUNTER["n"] == 3                        # 第一轮 3 次执行；第二轮 3 次「待执行」触发总量上限
    assert result.score == 0.82


# ==================== 5.5) contextvars 并发隔离 ====================

async def test_contextvars_isolate_references_concurrent():
    """并发评审时 contextvars 协程隔离：两个协程各看各的文献，不串数据。

    （设计 3.4：用全局变量会在并发章节写作/评审时串数据，必须用 contextvars）
    """
    async def verify_with_refs(refs, ref_index):
        with set_current_references(refs):
            await asyncio.sleep(0.01)   # 让出事件循环，制造交错
            return await verify_citation.ainvoke({"ref_index": ref_index, "claim": "任意论断"})

    refs_a = [{"title": "A1", "text": "a 正文"}]                              # 只有 1 条
    refs_b = [{"title": "B1", "text": "b 正文"}, {"title": "B2", "text": ""}]  # 2 条，B2 无 text

    a, b = await asyncio.gather(
        verify_with_refs(refs_a, 2),   # 协程 A：ref_index=2 越界
        verify_with_refs(refs_b, 2),   # 协程 B：ref_index=2 存在但无语料正文
    )
    assert a["verdict"] == "not_found" and "越界" in a["reason"]
    assert b["verdict"] == "not_found" and "无语料正文" in b["reason"]


# ==================== 5) 文献内检索工具 ====================

async def test_finalize_strips_unpaired_tool_calls(monkeypatch):
    """真实冒烟暴露的 bug：模型发起 tool_calls 后被预算强制收口，
    finalize 直接把「未配对的 tool_calls」发给 DeepSeek 会 400。
    收口前必须裁掉未完成的工具往返，只留干净对话。"""
    COUNTER["n"] = 0
    valid = AIMessage(content=VALID_REVIEW_JSON)
    # 剧本：3 条 tool_call 消息。前 2 轮在预算内（MAX_TOOL_ROUNDS=2）会执行，
    # 第 3 条触发轮数超限 → 强制 finalize。finalize 若不清洗，
    # 残留的「未配对 tool_calls + tool 结果」会让真实模型 400。
    model = ScriptedChatModel(script=[_tc(1), _tc(2), _tc(3), valid, valid, valid])
    result, _ = await run_reviewer(model, [("human", "请评审")], tools=[echo_probe])
    assert result.score == 0.82
    assert COUNTER["n"] == 2          # 前 2 轮工具执行；第 3 次被轮数上限强制收口


async def test_search_literature_indexes_are_citation_compatible(monkeypatch):
    """search_literature 返回的 index 必须与 verify_citation 同源（当前文献列表的编号）"""
    refs = [
        {"title": "RAG文献", "text": "检索增强生成通过外部知识库增强生成"},
        {"title": "BERT文献", "text": "预训练语言模型的掩码建模方法"},
    ]
    vectors = {
        "检索增强生成": [1.0, 0.0],
        "检索增强生成通过外部知识库增强生成": [0.9, 0.436],
        "预训练语言模型的掩码建模方法": [0.0, 1.0],
    }
    monkeypatch.setattr(citation, "_embed", lambda text: vectors[text])

    with set_current_references(refs):
        out = await search_literature.ainvoke({"query": "检索增强生成", "top_k": 3})
        assert out[0]["index"] == 1                    # 编号从 1 开始，对应 [1] 引用
        assert out[0]["title"] == "RAG文献"
        assert all(o["index"] != 2 for o in out)       # 无关文献（cos=0）被 0.3 阈值过滤

        # 空文献列表 → 空结果，不抛错
        with set_current_references([]):
            assert await search_literature.ainvoke({"query": "任意", "top_k": 3}) == []


# ==================== 6) 2026-11-01 端到端评测暴露的 3 个降级 bug ====================
# 真实评测第 1 篇降级，日志显示三级全败：
#   function_calling → KeyError('verify_citation')   模型幻觉调用工具
#   json_mode        → 400 Prompt must contain 'json' 收口指令不含 json
#   raw_parse        → dimensions 是 dict、缺 score   形状漂移判死
# 以下 4 项分别锁定修复。

def test_review_schema_tolerates_dict_dimensions():
    """模型把 dimensions 写成 {维度名: {...}} 且不给 score 时，应救回而非整篇降级。"""
    payload = {
        "dimensions": {
            "创新性与学术价值": {"score": 0.35, "comment": "缺乏创新"},
            "逻辑结构与论证": {"score": 0.45, "comment": "结构松散"},
            "内容完整度与充分性": {"score": 0.40, "comment": "实验不足"},
            "语言表达与学术规范": {"score": 0.55, "comment": "表达一般"},
            "文献引用质量": {"score": 0.50, "comment": "引用偏少"},
        },
        "overall": "整体存在明显缺陷",
    }
    r = ReviewResult.model_validate(payload)
    assert len(r.dimensions) == 5
    assert [d.dimension for d in r.dimensions][0] is Dimension.INNOVATION
    assert r.score == pytest.approx(0.45, abs=0.01)   # 缺 score → 用均值补
    # 维度名本身不合法时仍应拒绝（不编造维度）
    bad = dict(payload)
    bad["dimensions"] = {"随便一个名字": {"score": 0.5, "comment": "x"}}
    with pytest.raises(ValidationError):
        ReviewResult.model_validate(bad)


class _CaptureStructured:
    """记录 with_structured_output 收到的消息，然后抛错逼降级链继续。"""

    def __init__(self, sink, method):
        self.sink, self.method = sink, method

    async def ainvoke(self, messages):
        self.sink.append((self.method, list(messages)))
        raise RuntimeError("boom")


class _CaptureLLM:
    def __init__(self):
        self.sink = []

    def with_structured_output(self, schema, include_raw=False, method="function_calling", **kw):
        return _CaptureStructured(self.sink, method)

    async def ainvoke(self, messages):
        self.sink.append(("raw_parse", list(messages)))
        raise RuntimeError("boom")


async def test_json_mode_auto_appends_json_hint():
    """json_mode 这一级必须保证 prompt 含 'json'，否则 DeepSeek 必然 400。"""
    llm = _CaptureLLM()
    with pytest.raises(StructuredOutputError):
        await call_structured(llm, _Out, [HumanMessage(content="给个分数")], node="probe")

    modes = [m for m, _ in llm.sink]
    assert modes == ["function_calling", "json_mode", "raw_parse"]
    json_msgs = next(msgs for mode, msgs in llm.sink if mode == "json_mode")
    joined = " ".join(str(getattr(m, "content", "")) for m in json_msgs).lower()
    assert "json" in joined
    # 原始消息不能被污染：function_calling 那一次不该被追加提示
    fc_msgs = next(msgs for mode, msgs in llm.sink if mode == "function_calling")
    assert len(fc_msgs) == 1


def test_finalize_prompt_strips_tool_instructions():
    """收口 prompt 不能出现工具名，否则模型会幻觉调用它导致结构化解析 KeyError。"""
    from app.agents.reviewer_agent import ReviewerAgent

    a = ReviewerAgent(llm=FakeListChatModel(responses=["x"]))
    assert "verify_citation" in a.get_system_prompt()
    assert "search_literature" in a.get_system_prompt()
    assert "verify_citation" not in a.get_finalize_system_prompt()
    assert "search_literature" not in a.get_finalize_system_prompt()
    # 角色与维度必须保留（收口仍要按 5 维打分）
    assert "创新性与学术价值" in a.get_finalize_system_prompt()


def test_clean_messages_replaces_system_and_drops_tool_roundtrip():
    """收口清洗：替换 system（去工具说明）+ 裁掉 tool_calls/tool 消息往返。"""
    from app.agents.reviewer_graph import clean_messages_for_finalize

    msgs = [
        SystemMessage(content="你可以使用 verify_citation 查证"),
        HumanMessage(content="请评审"),
        AIMessage(content="", tool_calls=[{"name": "verify_citation", "args": {}, "id": "c1"}]),
        ToolMessage(content="supported", tool_call_id="c1"),
        AIMessage(content="我的评审意见……"),
    ]
    out = clean_messages_for_finalize(msgs, system_prompt="干净的评审提示")
    assert out[0].type == "system" and out[0].content == "干净的评审提示"
    assert all(getattr(m, "type", None) != "tool" for m in out)
    assert not any(getattr(m, "tool_calls", None) for m in out)
    # 纯文本 AI 结论保留（可走 finalize 快路径）
    assert any(getattr(m, "content", "") == "我的评审意见……" for m in out)


# ==================== 7) 模型输出上限 ====================

def test_llm_sets_max_tokens_explicitly():
    """不显式设 max_tokens 会落到 DeepSeek 默认 4096，长评审 JSON 被截断成非法 JSON，
    三级降级全失败导致整篇评审作废（2026-11-02 实测）。这里锁死：必须显式设置。"""
    from app.config import settings
    from app.utils.llm import get_llm

    llm = get_llm(temperature=0.0)
    assert llm.max_tokens is not None
    assert llm.max_tokens == settings.LLM_MAX_TOKENS >= 8192

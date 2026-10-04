# 单元测试：6 个 Agent 的 execute()（全部用假 LLM / 假检索器，不联网、不花 token）
import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.runnables import RunnableLambda

from app.agents.outline_agent import OutlineAgent
from app.agents.polish_agent import PolishAgent
from app.agents.reference_agent import ReferenceAgent
from app.agents.reviewer_agent import ReviewerAgent
from app.agents.topic_agent import TopicAgent
from app.agents.writer_agent import WriterAgent


def _with_llm(agent, *responses):
    """把 Agent 的大脑换成假模型（按顺序返回给定文本）"""
    agent.llm = FakeListLLM(responses=list(responses))
    return agent


def _capture(store):
    """假模型：把渲染后的 Prompt 文本记到 store 里，再返回固定文本"""
    def _echo(prompt_value):
        store["text"] = "".join(
            m.content if isinstance(m.content, str) else str(m.content)
            for m in prompt_value.messages
        )
        return "模型输出"
    return RunnableLambda(_echo)


# ================= 选题 Agent =================

async def test_topic_agent_returns_markdown_topics():
    a = _with_llm(TopicAgent(), "## 选题推荐\n### 选题 1：RAG 幻觉缓解")
    out = await a.execute({"field": "人工智能", "keywords": "RAG", "requirements": ""})
    assert "RAG 幻觉缓解" in out["topics"]


async def test_topic_agent_puts_requirements_into_prompt():
    store = {}
    a = TopicAgent()
    a.llm = _capture(store)
    await a.execute({"field": "NLP", "keywords": "检索", "requirements": "偏工程落地"})
    assert "特殊要求：偏工程落地" in store["text"]
    assert "检索" in store["text"]


async def test_topic_agent_stream_execute_yields_chunks():
    a = _with_llm(TopicAgent(), "第一段")
    chunks = [c async for c in a.stream_execute({"field": "AI", "keywords": ""})]
    assert chunks == ["第一段"]


# ================= 大纲 Agent =================

async def test_outline_agent_parses_json_into_dict():
    payload = '```json\n{"title": "T", "sections": [{"title": "引言", "points": ["背景"], "word_count": 800}]}\n```'
    a = _with_llm(OutlineAgent(), payload)
    out = await a.execute({"title": "T", "paper_type": "research", "word_limit": 6000})
    assert out["outline_dict"]["sections"][0]["title"] == "引言"


async def test_outline_agent_invalid_json_yields_empty_dict_but_keeps_raw():
    a = _with_llm(OutlineAgent(), "抱歉我无法输出 JSON")
    out = await a.execute({"title": "T", "paper_type": "research", "word_limit": 6000})
    # 解析失败 → outline_dict 退化成 {"raw": 原文}，不会崩；但绝不能凭空造出 sections
    assert out["outline_dict"]["raw"] == "抱歉我无法输出 JSON"
    assert "sections" not in out["outline_dict"]
    assert out["outline"] == "抱歉我无法输出 JSON"


# ================= 写作 Agent =================

async def test_writer_agent_returns_content():
    a = _with_llm(WriterAgent(), "这是章节正文。")
    out = await a.execute({"section_title": "引言", "outline_points": ["背景"], "word_count": 300})
    assert out["content"] == "这是章节正文。"


async def test_writer_prompt_numbers_reference_list():
    """list 入参 → 按 [1][2] 编号渲染进 Prompt"""
    store = {}
    a = WriterAgent()
    a.llm = _capture(store)
    await a.execute({
        "section_title": "方法", "outline_points": ["方案"], "word_count": 200,
        "references": ["作者甲. 论文A[J]. 刊. 2020.", "作者乙. 论文B[J]. 刊. 2021."],
    })
    assert "[1] 作者甲. 论文A[J]. 刊. 2020." in store["text"]
    assert "[2] 作者乙. 论文B[J]. 刊. 2021." in store["text"]
    assert "禁止编造" in store["text"]


async def test_writer_accepts_prenumbered_string_references():
    """str 入参（已编号整段文本）→ 原样注入，不重复编号"""
    store = {}
    a = WriterAgent()
    a.llm = _capture(store)
    await a.execute({
        "section_title": "方法", "outline_points": [], "word_count": 100,
        "references": "[1] 张三. 论文A[J].  NeurIPS. 2020.",
    })
    assert store["text"].count("[1] 张三. 论文A[J].") == 1
    assert "[2]" not in store["text"]


async def test_writer_prompt_falls_back_when_no_references():
    """无文献时必须出现「暂无参考文献」占位，而不是空白或虚构编号"""
    store = {}
    a = WriterAgent()
    a.llm = _capture(store)
    await a.execute({"section_title": "结论", "outline_points": [], "word_count": 100})
    assert "暂无参考文献" in store["text"]


async def test_writer_prompt_carries_feedback_on_revision():
    store = {}
    a = WriterAgent()
    a.llm = _capture(store)
    await a.execute({
        "section_title": "方法", "outline_points": [], "word_count": 100,
        "feedback": "论证不足，请补充对比实验",
    })
    assert "论证不足，请补充对比实验" in store["text"]

    # 首次撰写：必须用占位文案，而不是空字符串
    store.clear()
    await a.execute({"section_title": "引言", "outline_points": [], "word_count": 100})
    assert "（首次撰写，暂无历史意见）" in store["text"]


async def test_writer_stream_execute_yields_chunks():
    a = _with_llm(WriterAgent(), "流式正文")
    chunks = [c async for c in a.stream_execute({"section_title": "引言", "outline_points": []})]
    assert chunks == ["流式正文"]


# ================= 润色 Agent =================

async def test_polish_agent_reports_both_lengths():
    a = _with_llm(PolishAgent(), "润色后")
    out = await a.execute({"content": "原始内容很长", "focus": "language"})
    assert out["polished"] == "润色后"
    assert out["original_length"] == len("原始内容很长")
    assert out["polished_length"] == len("润色后")


async def test_polish_agent_unknown_focus_falls_back_to_all():
    store = {}
    a = PolishAgent()
    a.llm = _capture(store)
    await a.execute({"content": "x", "focus": "不存在的侧重"})
    assert "请全面优化语言表达、逻辑结构和格式规范。" in store["text"]


async def test_polish_agent_stream_execute_yields_chunks():
    a = _with_llm(PolishAgent(), "流式润色")
    chunks = [c async for c in a.stream_execute({"content": "x"})]
    assert chunks == ["流式润色"]


# ================= 文献 Agent =================

class _FakeRetriever:
    """假检索器：记录调用参数，返回预设文献"""

    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    async def search(self, query, top_k=5):
        self.calls.append({"query": query, "top_k": top_k})
        return list(self.docs)


def _docs():
    return [
        {"id": "ref-010", "title": "RAG", "authors": "Lewis 等", "journal": "NeurIPS", "year": 2020},
        {"id": "ref-002", "title": "BERT", "authors": "Devlin 等", "journal": "NAACL", "year": 2019},
    ]


async def test_reference_agent_retrieve_only_searches():
    agent = ReferenceAgent()
    agent.retriever = _FakeRetriever(_docs())
    docs = await agent.retrieve("检索增强生成", top_k=5)
    assert [d["id"] for d in docs] == ["ref-010", "ref-002"]
    assert agent.retriever.calls == [{"query": "检索增强生成", "top_k": 5}]


async def test_reference_agent_execute_formats_with_llm():
    store = {}
    agent = ReferenceAgent()
    agent.retriever = _FakeRetriever(_docs())
    agent.llm = _capture(store)

    out = await agent.execute({"query": "RAG", "top_k": 2})
    assert out["references"] == _docs()          # 原始检索结果原样返回
    assert out["formatted"] == "模型输出"
    # 元信息被拼成文本喂给了模型
    assert "标题: RAG" in store["text"] and "年份: 2020" in store["text"]


async def test_reference_agent_stream_execute_yields_chunks():
    agent = ReferenceAgent()
    agent.retriever = _FakeRetriever(_docs())
    agent.llm = FakeListLLM(responses=["引用列表"])
    chunks = [c async for c in agent.stream_execute({"query": "RAG"})]
    assert chunks == ["引用列表"]


def test_format_citation_strings_is_deterministic():
    strings = ReferenceAgent.format_citation_strings(_docs())
    assert strings == [
        "Lewis 等. RAG[J]. NeurIPS. 2020.",
        "Devlin 等. BERT[J]. NAACL. 2019.",
    ]


def test_format_citation_strings_degrades_on_missing_fields():
    assert ReferenceAgent.format_citation_strings([{"title": "C"}]) == ["佚名. C[J]."]
    assert ReferenceAgent.format_citation_strings([]) == []
    assert ReferenceAgent.format_citation_strings(None) == []


def test_format_gbt_numbering_matches_string_order():
    """编号顺序必须与 format_citation_strings 的列表顺序严格一致"""
    assert ReferenceAgent.format_gbt(_docs()) == (
        "[1] Lewis 等. RAG[J]. NeurIPS. 2020.\n"
        "[2] Devlin 等. BERT[J]. NAACL. 2019."
    )


# ================= 评审 Agent =================

VALID_REVIEW = """```json
{
  "score": 0.82,
  "dimensions": [
    {"dimension": "创新性与学术价值", "score": 0.8, "comment": "有一定创新"},
    {"dimension": "逻辑结构与论证", "score": 0.85, "comment": "论证清晰"},
    {"dimension": "内容完整度与充分性", "score": 0.8, "comment": "内容完整"},
    {"dimension": "语言表达与学术规范", "score": 0.83, "comment": "语言规范"},
    {"dimension": "文献引用质量", "score": 0.8, "comment": "引用恰当"}
  ],
  "overall": "总体不错，建议补充对比实验。"
}
```"""


async def test_reviewer_parses_valid_json():
    a = _with_llm(ReviewerAgent(), VALID_REVIEW)
    result = await a.execute({"title": "T", "topic": "NLP", "content": "正文", "references": "文献"})
    data = result["data"]
    assert data["score"] == 0.82
    assert len(data["dimensions"]) == 5
    assert data["dimensions"][0]["dimension"] == "创新性与学术价值"
    assert "对比实验" in data["overall"]


async def test_reviewer_handles_invalid_json():
    """非法 JSON 不崩溃，data 里带原始文本"""
    a = _with_llm(ReviewerAgent(), "这不是 JSON 输出")
    result = await a.execute({"title": "T", "topic": "NLP", "content": "C", "references": "R"})
    assert result["data"]["raw"] == "这不是 JSON 输出"


async def test_reviewer_handles_empty_response():
    a = _with_llm(ReviewerAgent(), "")
    result = await a.execute({"title": "T", "topic": "NLP", "content": "C", "references": "R"})
    assert result["data"] == {}


async def test_reviewer_prompt_contains_paper_fields():
    store = {}
    a = ReviewerAgent()
    a.llm = _capture(store)
    await a.execute({"title": "题目X", "topic": "方向Y", "content": "内容Z", "references": "文献W"})
    for token in ("题目X", "方向Y", "内容Z", "文献W"):
        assert token in store["text"]

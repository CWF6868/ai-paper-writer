# test_demo_mode.py —— 演示模式（DEMO_MODE）：零外呼可跑通，且不绕开任何校验
#
# 这一层最容易出的问题不是"跑不起来"，而是"看起来跑起来了"：
#   - with_structured_output 返回形状不对 → 三级降级全失败 → 整篇降级，但页面照样有内容；
#   - 评审解析拿错消息 → 分数与分章建议失真，却仍是一个合法分数；
#   - 引用编号凭空生成 → 演示里出现不存在的文献。
# 因此下面的断言都盯着这几点，而不是只看"有没有返回"。
import re

import pytest
from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.core.llm_client import build_chat_model, build_embedding_model
from app.demo.chat import DemoChatModel, DemoHashEmbeddings, detect_node, reply_text
from app.demo.content import build_outline, build_review, build_section
from app.llm.schemas import OutlineResult, ReviewResult
from app.llm.structured import call_structured
from app.utils.llm import check_llm_config


# ---------------------------------------------------------------------------
# 节点识别：靠人设关键词，而不是调用顺序
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("persona,node", [
    ("你是一位资深学术论文大纲专家", "outline"),
    ("你是一位严谨的学术论文评审专家", "reviewer"),
    ("你是一位资深的学术论文写作专家", "writer"),
    ("你是一位资深的学术论文润色专家", "polish"),
    ("你是一位资深的学术研究选题专家", "topic"),
    ("你是一位资深的学术文献专家", "reference"),
    ("其它人设", "generic"),
])
def test_detect_node_by_persona(persona, node):
    messages = [SystemMessage(content=persona), HumanMessage(content="随便")]
    assert detect_node(messages) == node


def test_reviewer_turn_returns_prose_not_json():
    """评审 agent 轮必须给自然语言。

    如果这里直接吐 JSON，finalize 的快路径会命中 raw_parse，
    真正的结构化收口（function_calling）就永远走不到 —— 演示看起来正常，
    实际把 P0-1 的降级链路藏起来了。
    """
    messages = [
        SystemMessage(content="你是一位严谨的学术论文评审专家"),
        HumanMessage(content="## 论文标题\n测试\n\n## 正文内容\n## 引言\n正文\n\n## 参考文献\n[1] x"),
    ]
    text = reply_text(messages)
    assert "{" not in text
    assert text.strip()


# ---------------------------------------------------------------------------
# 结构化输出：必须走 function_calling 这一级（不是降级）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_outline_structured_hits_function_calling():
    llm = DemoChatModel()
    messages = [
        SystemMessage(content="你是一位资深学术论文大纲专家"),
        HumanMessage(content="请为下面的论文设计大纲：\n标题：检索增强生成研究\n论文类型：research\n总字数要求：约 6000 字"),
    ]
    result, mode = await call_structured(llm, OutlineResult, messages, node="outline")

    assert mode == "function_calling"
    assert result.title == "检索增强生成研究"
    assert [s.title for s in result.sections] == ["摘要", "引言", "相关工作", "方法", "实验与分析", "结论"]
    assert all(len(s.points) >= 2 for s in result.sections)


@pytest.mark.asyncio
async def test_review_structured_hits_function_calling_and_reads_full_prompt():
    """收口时最后一条 human 是「请输出评审结果」这句指令。

    评审对象必须从更早的那条长消息里取；只取最后一条会让分章建议为空、
    引用维度被误判成"没有引用"。
    """
    llm = DemoChatModel()
    review_prompt = (
        "请评审下面这篇论文。\n\n## 论文标题\n检索增强生成研究\n\n## 研究方向\nRAG\n\n"
        "## 正文内容\n## 引言\n本文讨论 RAG [1]。\n\n## 方法\n方法描述 [2]。\n\n"
        "## 参考文献\n[1] A. X[J]. 2020.\n[2] B. Y[J]. 2021."
    )
    messages = [
        SystemMessage(content="你是一位严谨的学术论文评审专家"),
        HumanMessage(content=review_prompt),
        HumanMessage(content="请基于上述评审材料，直接输出符合要求的评审结果。"),
    ]
    result, mode = await call_structured(llm, ReviewResult, messages, node="reviewer_finalize")

    assert mode == "function_calling"
    assert 0.0 <= result.score <= 1.0
    assert len(result.dimensions) == 5
    # 读到的是完整 Prompt → 能给出分章建议，而不是空列表
    assert [fb.section for fb in result.section_feedback] == ["引言", "方法"]
    # must_fix 为空，演示不会触发重写循环
    assert result.must_fix == []


def test_review_score_above_default_gate():
    """演示分要高于默认门禁 0.44，否则每次演示都进重写循环，3 分钟变 8 分钟。"""
    prompt = (
        "## 论文标题\n检索增强生成研究\n\n## 正文内容\n"
        + ("## 引言\n讨论检索增强生成的有效性。[1]\n\n" * 20)
        + "\n## 参考文献\n[1] A. X[J]. 2020."
    )
    review = ReviewResult.model_validate(build_review(prompt))
    assert review.score >= settings.QUALITY_PASS_THRESHOLD


def test_review_without_references_lowers_citation_dimension():
    """没有参考文献时引用维度必须低分，不能"没引用也给高分"。"""
    prompt = "## 论文标题\n测试\n\n## 正文内容\n## 引言\n没有任何引用。\n\n## 参考文献\n暂无参考文献"
    review = ReviewResult.model_validate(build_review(prompt))
    citation = [d for d in review.dimensions if d.dimension.value == "文献引用质量"][0]
    assert citation.score < 0.6


# ---------------------------------------------------------------------------
# 正文写作：引用只能来自给定列表
# ---------------------------------------------------------------------------
def _writer_prompt(refs: str) -> str:
    return (
        "请撰写以下章节内容：\n\n## 章节标题：方法\n\n### 内容要点：\n- 整体框架设计\n- 检索与证据注入策略\n\n"
        f"### 相关文献（本节可引用的全部文献，编号固定，不得改写）：\n{refs}\n\n"
        "### 历史评审改进建议：\n（首次撰写，暂无历史意见）\n\n### 要求：\n- 写作风格：formal"
    )


def test_section_uses_only_given_citation_numbers():
    text = build_section(_writer_prompt("[1] A. X[J]. 2020.\n[2] B. Y[J]. 2021."))
    assert "[1]" in text and "[2]" in text
    assert "[3]" not in text


def test_section_without_references_has_no_citation_mark():
    """列表为空时任何 [n] 都是编造 —— 这是引用铁律，演示模式也不能破。"""
    text = build_section(_writer_prompt("暂无参考文献"))
    assert re.search(r"\[\d+\]", text) is None


def test_section_is_deterministic_and_non_trivial():
    prompt = _writer_prompt("暂无参考文献")
    a, b = build_section(prompt), build_section(prompt)
    assert a == b
    assert len(a) > 200


# ---------------------------------------------------------------------------
# 向量化：确定性 + 真的能排序
# ---------------------------------------------------------------------------
def test_embeddings_are_deterministic_across_instances():
    """用 md5 而不是内置 hash：跨进程重算必须一致，否则灌库与查询对不上。"""
    text = "检索增强生成缓解大模型幻觉"
    v1 = DemoHashEmbeddings().embed_query(text)
    v2 = DemoHashEmbeddings().embed_query(text)
    assert v1 == v2
    assert len(v1) == 256


def test_embeddings_rank_related_text_higher():
    emb = DemoHashEmbeddings()
    query = emb.embed_query("检索增强生成")
    related = emb.embed_query("检索增强生成通过外部知识缓解幻觉")
    unrelated = emb.embed_query("图像分割中的注意力机制")

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    assert cos(query, related) > cos(query, unrelated)


def test_embed_documents_batch_matches_single():
    emb = DemoHashEmbeddings()
    docs = ["检索增强生成", "图神经网络"]
    assert emb.embed_documents(docs) == [emb.embed_query(d) for d in docs]


# ---------------------------------------------------------------------------
# 构造入口与配置校验
# ---------------------------------------------------------------------------
def test_build_models_switch_to_demo(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    assert isinstance(build_chat_model(), DemoChatModel)
    assert isinstance(build_embedding_model(), DemoHashEmbeddings)


def test_demo_seeding_switches_to_dedicated_collection(monkeypatch):
    """演示向量是 256 维、正式库是 1024 维，必须在各自集合里，否则 Chroma 报维度不匹配。

    本机踩过：data/chroma 里已有真实 1024 维集合时，演示模式直接 500。
    """
    from app.demo import seed as seed_module
    from app.rag.vector_store import COLLECTION_NAME, DEMO_COLLECTION_NAME, vector_store

    monkeypatch.setattr(settings, "DEMO_MODE", True)
    seed_module.configure_demo_embeddings()
    assert vector_store.collection_name == DEMO_COLLECTION_NAME
    assert vector_store.collection.name == DEMO_COLLECTION_NAME

    # 切回正式集合后仍能拿到原来的集合，不能把生产数据指向演示集合
    vector_store.use_collection(COLLECTION_NAME)
    assert vector_store.collection.name == COLLECTION_NAME


def test_check_llm_config_skips_in_demo(monkeypatch):
    """演示实例本来就没有 Key，这里拦一次就把 Demo 拦成 400。"""
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", None)
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", None)
    check_llm_config()          # 不抛异常即通过


def test_check_llm_config_still_blocks_when_not_demo(monkeypatch):
    """非演示模式必须继续拦截 —— 否则真实环境会带着空 Key 去请求。"""
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", None)
    with pytest.raises(HTTPException) as exc:
        check_llm_config()
    assert exc.value.status_code == 400


def test_bind_tools_does_not_mutate_original():
    """bind_tools 必须返回副本：原实例被改脏，后续节点的行为会跟着变。"""
    from langchain_core.tools import tool

    @tool
    def verify_citation(ref_index: int, claim: str) -> str:
        """查证引用"""
        return "ok"

    base = DemoChatModel()
    bound = base.bind_tools([verify_citation])
    assert bound.bound_tool_names == ("verify_citation",)
    assert base.bound_tool_names == ()


@pytest.mark.asyncio
async def test_stream_yields_content():
    model = DemoChatModel()
    messages = [SystemMessage(content="你是一位资深的学术研究选题专家"), HumanMessage(content="研究领域：RAG")]
    chunks = [c.content async for c in model.astream(messages)]
    assert "".join(chunks).strip()


def test_unknown_schema_raises_instead_of_faking():
    """未覆盖的 Schema 必须显式报错，不能返回一个看起来合法的空对象。"""
    from app.demo.chat import _schema_payload

    class Other:
        pass

    Other.__name__ = "OtherSchema"
    with pytest.raises(ValueError):
        _schema_payload(Other, [HumanMessage(content="x")])


def test_build_outline_echoes_title_from_prompt():
    assert build_outline("标题：某个具体选题\n论文类型：research")["title"] == "某个具体选题"
    assert build_outline("纯标题")["title"] == "纯标题"


# ---------------------------------------------------------------------------
# 数据初始化：幂等 + 不联网
# ---------------------------------------------------------------------------
def test_configure_demo_embeddings_swaps_embedding_and_seeds(monkeypatch):
    """即使向量库单例已经被真实 embedding 初始化过，也必须换成本地实现再灌库。

    不换的话，灌库会真的打向量化接口 —— "零外呼"的承诺当场破掉。
    """
    from app.demo.seed import configure_demo_embeddings
    from app.rag.corpus import ACADEMIC_REFERENCES

    class FakeStore:
        def __init__(self):
            self.embeddings = "real-embedding"
            self.docs = []
            self.collection_name = "academic_references"

        def use_collection(self, name):
            self.collection_name = name

        def get_document_count(self):
            return len(self.docs)

        def add_documents(self, docs):
            self.docs.extend(docs)

    monkeypatch.setattr("app.rag.vector_store.vector_store", FakeStore())
    monkeypatch.setattr(settings, "DEMO_MODE", True)

    from app.rag.vector_store import DEMO_COLLECTION_NAME
    from app.rag.vector_store import vector_store as fake
    count = configure_demo_embeddings()

    assert count == len(ACADEMIC_REFERENCES)
    assert isinstance(fake.embeddings, DemoHashEmbeddings)
    assert fake.collection_name == DEMO_COLLECTION_NAME


@pytest.mark.asyncio
async def test_ensure_demo_data_skipped_when_demo_mode_off(monkeypatch):
    from app.demo.seed import ensure_demo_data

    monkeypatch.setattr(settings, "DEMO_MODE", False)
    assert (await ensure_demo_data())["skipped"] is True


@pytest.mark.asyncio
async def test_ensure_demo_user_is_idempotent(monkeypatch):
    """重复启动不能重复建账号：演示实例会被反复重启，每重启一次多一个用户很荒唐。"""
    from sqlalchemy import func, select

    from app.database import AsyncSessionLocal
    from app.demo import seed as seed_module
    from app.models.user import User

    monkeypatch.setattr(settings, "DEMO_MODE", True)
    monkeypatch.setattr(seed_module, "configure_demo_embeddings", lambda: 42)

    first = await seed_module.ensure_demo_data(seed_paper=False)
    second = await seed_module.ensure_demo_data(seed_paper=False)

    assert first["user_email"] == second["user_email"] == seed_module.DEMO_EMAIL
    async with AsyncSessionLocal() as db:
        total = (await db.execute(
            select(func.count()).select_from(User).where(User.email == seed_module.DEMO_EMAIL)
        )).scalar_one()
    assert total == 1


@pytest.mark.asyncio
async def test_ensure_sample_paper_generates_once(monkeypatch):
    """演示论文必须真的经由工作流产出入库，且重复初始化不重复建。

    用假工作流替身只是为了不在单测里跑完整流水线；
    真正跑通流水线的证据在冒烟脚本里（scripts/smoke_demo.py）。
    """
    from sqlalchemy import func, select

    from app.database import AsyncSessionLocal
    from app.demo import seed as seed_module
    from app.models.paper import Paper

    class FakeWorkflow:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, state):
            return {
                "outline": {"sections": [{"title": "引言", "points": [], "word_count": 600}]},
                "sections_content": {"引言": "正文内容"},
                "final_paper": "## 引言\n\n正文内容",
                "references": [{"title": "某文献", "authors": "张三", "journal": "某刊", "year": 2020}],
                "quality_score": 0.7,
                "quality_comment": "结构完整",
                "quality_dimensions": [],
                "quality_degraded": False,
                "best_score": 0.7,
                "best_paper": "## 引言\n\n正文内容",
                "best_sections": {"引言": "正文内容"},
            }

    monkeypatch.setattr(settings, "DEMO_MODE", True)
    monkeypatch.setattr(seed_module, "configure_demo_embeddings", lambda: 42)
    monkeypatch.setattr("app.workflows.graph_builder.PaperWorkflowGraph", FakeWorkflow)

    first = await seed_module.ensure_demo_data()
    second = await seed_module.ensure_demo_data()

    assert first["paper_id"] == second["paper_id"]
    assert first["paper_status"] == "completed"
    async with AsyncSessionLocal() as db:
        total = (await db.execute(select(func.count()).select_from(Paper))).scalar_one()
    assert total == 1


@pytest.mark.asyncio
async def test_empty_sample_paper_gets_regenerated(monkeypatch):
    """初始化中途失败会留下一条空稿；幂等逻辑必须补跑，不能永远返回空论文。

    真实踩点：向量库维度不匹配导致首次生成失败，重启后 _ensure_sample_paper
    看到记录已存在就直接返回，演示首屏于是永远是一篇 0 字论文。
    """
    from sqlalchemy import select

    from app.database import AsyncSessionLocal
    from app.demo import seed as seed_module
    from app.models.paper import Paper, PaperStatus

    monkeypatch.setattr(settings, "DEMO_MODE", True)
    monkeypatch.setattr(seed_module, "DEMO_PAPER_TITLE", "空稿重跑用例")

    async with AsyncSessionLocal() as db:
        user = await seed_module._ensure_user(db)
        db.add(Paper(
            title="空稿重跑用例", topic="t", keywords="k",
            author_id=user.id, status=PaperStatus.DRAFT,
        ))
        await db.commit()

    calls = []

    class FakeWorkflow:
        def __init__(self, *a, **kw):
            pass

        async def run(self, state):
            calls.append(state)
            return {"sections_content": {"引言": "补跑出来的正文"}}

    async def fake_save(paper, state, db):
        paper.content = "补跑出来的正文"
        await db.commit()

    monkeypatch.setattr("app.workflows.graph_builder.PaperWorkflowGraph", FakeWorkflow)
    monkeypatch.setattr("app.api.papers._save_results", fake_save)

    async with AsyncSessionLocal() as db:
        paper = await seed_module._ensure_sample_paper(db, user)
        pid = paper.id
    assert paper.content == "补跑出来的正文"
    assert len(calls) == 1

    # 已经有正文之后不能再跑第二遍（否则每次重启都重生成、重复计费）
    async with AsyncSessionLocal() as db:
        again = await seed_module._ensure_sample_paper(db, user)
    assert again.id == pid
    assert len(calls) == 1
    async with AsyncSessionLocal() as db:
        titles = (await db.execute(
            select(Paper).where(Paper.title == "空稿重跑用例")
        )).scalars().all()
    assert len(titles) == 1


# ---------------------------------------------------------------------------
# 其余节点的文本输出（选题 / 文献 / 润色 / 兜底）
# ---------------------------------------------------------------------------
def test_topic_reply_contains_three_options():
    messages = [
        SystemMessage(content="你是一位资深的学术研究选题专家"),
        HumanMessage(content="研究领域：检索增强生成"),
    ]
    text = reply_text(messages)
    assert "选题 1" in text and "选题 3" in text
    assert "检索增强生成" in text


def test_reference_reply_lists_only_given_titles():
    messages = [
        SystemMessage(content="你是一位资深的学术文献专家"),
        HumanMessage(content="文献 1:\n标题: Attention Is All You Need\n作者: Vaswani\n\n文献 2:\n标题: BERT\n作者: Devlin"),
    ]
    text = reply_text(messages)
    assert text.splitlines()[0].startswith("[1]")
    assert "Attention" in text


def test_reference_reply_without_documents_says_none():
    messages = [
        SystemMessage(content="你是一位资深的学术文献专家"),
        HumanMessage(content="检索主题：空"),
    ]
    assert reply_text(messages) == "暂无参考文献"


def test_polish_reply_returns_source_text_unchanged():
    """润色必须原样返回正文：演示模式不伪造改写效果，也保证不会把正文写丢。"""
    body = "## 引言\n\n这是一段需要润色的正文，包含引用 [1]。"
    messages = [
        SystemMessage(content="你是一位资深的学术论文润色专家"),
        HumanMessage(content=f"请润色以下论文内容：\n\n请全面优化。\n\n## 原文内容：\n\n{body}\n\n## 输出要求：\n\n只输出润色后的正文。"),
    ]
    assert reply_text(messages).strip() == body


def test_generic_reply_is_explicit_about_demo():
    text = reply_text([HumanMessage(content="你好")])
    assert "演示模式" in text


def test_outline_text_fallback_is_valid_json():
    messages = [
        SystemMessage(content="你是一位资深学术论文大纲专家"),
        HumanMessage(content="标题：回退标题"),
    ]
    import json

    payload = json.loads(reply_text(messages))
    assert OutlineResult.model_validate(payload).title == "回退标题"


# ---------------------------------------------------------------------------
# 结构化输出的两条分支 + 引文裁决
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_structured_output_without_raw_returns_model():
    llm = DemoChatModel()
    structured = llm.with_structured_output(OutlineResult, include_raw=False)
    result = await structured.ainvoke([HumanMessage(content="标题：直接解析")])
    assert isinstance(result, OutlineResult)


@pytest.mark.asyncio
async def test_structured_output_unknown_schema_raises():
    """未覆盖 Schema 必须抛错（上层的三级降级会接住），不能静默返回假对象。"""
    from langchain_core.runnables import RunnableLambda

    class Other:
        pass

    Other.__name__ = "OtherSchema"
    llm = DemoChatModel()
    structured = llm.with_structured_output(Other, include_raw=False)
    assert isinstance(structured, RunnableLambda)
    with pytest.raises(ValueError):
        await structured.ainvoke([HumanMessage(content="x")])


@pytest.mark.parametrize("sim,verdict", [
    (0.9, "supported"),
    (0.5, "partial"),
    (0.1, "unsupported"),
])
def test_citation_verdict_bands(sim, verdict):
    from app.demo.chat import _schema_payload
    from app.llm.schemas import CitationVerdict

    payload = _schema_payload(
        CitationVerdict,
        [HumanMessage(content=f"参考文献正文片段：...\n（向量初筛相似度 {sim:.2f}，仅供参考）")],
    )
    assert CitationVerdict.model_validate(payload).verdict == verdict


def test_embedding_of_empty_text_is_zero_vector():
    """空串不能炸：Chroma 查询可能带空 query（论文标题为空的场景）。"""
    vec = DemoHashEmbeddings().embed_query("")
    assert len(vec) == 256 and all(x == 0.0 for x in vec)


def test_review_prompt_without_sections_has_empty_feedback():
    prompt = "## 论文标题\n无章节\n\n## 正文内容\n一段没有二级标题的正文。[1]\n\n## 参考文献\n[1] A."
    review = ReviewResult.model_validate(build_review(prompt))
    assert review.section_feedback == []


# ---------------------------------------------------------------------------
# 边界：消息形态、缺失人设、同步流式
# ---------------------------------------------------------------------------
def test_message_text_handles_content_blocks():
    """LangChain 的 content 可能是 block 列表（多模态/新版消息），不能直接当字符串用。"""
    from langchain_core.messages import AIMessage

    from app.demo.chat import _message_text

    assert _message_text(AIMessage(content=[{"type": "text", "text": "片段"}])) == "片段"


def test_missing_persona_and_missing_human_do_not_crash():
    from app.demo.chat import _last_human_text

    assert detect_node([HumanMessage(content="只有人类消息")]) == "generic"
    assert _last_human_text([SystemMessage(content="只有系统消息")]) == ""


def test_sync_stream_yields_single_chunk():
    model = DemoChatModel()
    messages = [SystemMessage(content="你是一位资深的学术研究选题专家"), HumanMessage(content="研究领域：RAG")]
    chunks = list(model.stream(messages))
    assert "".join(c.content for c in chunks).strip()


@pytest.mark.asyncio
async def test_structured_output_reports_parsing_error_instead_of_raising():
    """include_raw=True 时，校验失败要放在 parsing_error 里交回上层，而不是直接抛。

    structured.py 正是靠这个字段决定继续降级；直接抛会让它把异常当成"该模式失败"，
    行为虽然也对，但拿不到原始输出，排障时看不到模型到底吐了什么。
    """
    class FakeSchema:
        """名称对得上 OutlineResult，但字段完全不同 —— 必然校验失败。"""

        @classmethod
        def model_validate(cls, payload):
            raise ValueError("字段不匹配")

    FakeSchema.__name__ = "OutlineResult"
    llm = DemoChatModel()
    out = await llm.with_structured_output(FakeSchema, include_raw=True).ainvoke(
        [HumanMessage(content="标题：x")]
    )
    assert out["parsed"] is None
    assert out["parsing_error"] is not None
    assert '"title": "x"' in out["raw"].content

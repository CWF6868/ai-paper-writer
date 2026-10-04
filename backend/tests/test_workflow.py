# 单元测试：workflows/graph_builder.py
#
# 分三层：
#   1) 条件路由函数（纯逻辑，不碰 Agent）
#   2) 各节点的兜底 / 分支逻辑
#   3) 用假 Agent 跑整张图（不联网、不花 token），锁死 RAG 注入链路
from app.workflows.graph_builder import PaperWorkflowGraph

FAKE_OUTLINE = {"sections": [
    {"title": "引言", "points": ["背景介绍"], "word_count": 200},
    {"title": "方法", "points": ["技术方案"], "word_count": 300},
]}


def _graph():
    return PaperWorkflowGraph()


def _base_state(**over):
    state = {
        "paper_id": 1, "title": "测试论文标题", "topic": "人工智能",
        "keywords": ["AI"], "is_topic_clear": True,
        "revision_count": 0, "max_revisions": 2, "sections_content": {},
    }
    state.update(over)
    return state


def _patch_agents(g, *, outline=None, retrieve=None, writer=None,
                  refs_execute=None, polish=None, review=None, topic=None):
    """把图上的 6 个 Agent 换成假的（只替换传了的那几个）"""
    if topic is not None:
        g.topic_agent.execute = topic
    if outline is not None:
        g.outline_agent.execute = outline
    if retrieve is not None:
        g.reference_agent.retrieve = retrieve
    if writer is not None:
        g.writer_agent.execute = writer
    if refs_execute is not None:
        g.reference_agent.execute = refs_execute
    if polish is not None:
        g.polish_agent.execute = polish
    if review is not None:
        g.reviewer_agent.execute = review
    return g


# ==================== 1) 条件路由 ====================

def test_check_quality_pass_on_high_score():
    g = _graph()
    assert g._check_quality({"quality_score": 0.7}) == "pass"
    assert g._check_quality({"quality_score": 0.95}) == "pass"


def test_check_quality_revise_then_exhausted():
    g = _graph()
    assert g._check_quality({"quality_score": 0.5, "revision_count": 0, "max_revisions": 2}) == "revise"
    # 次数用尽 → 放行结束（防死循环）
    assert g._check_quality({"quality_score": 0.5, "revision_count": 2, "max_revisions": 2}) == "pass"


def test_check_quality_missing_score_defaults_to_zero():
    g = _graph()
    assert g._check_quality({}) == "revise"


def test_should_recommend_topic_routing():
    g = _graph()
    assert g._should_recommend_topic({"is_topic_clear": True}) == "outline"
    assert g._should_recommend_topic({"is_topic_clear": False}) == "recommend"
    # 已推荐过一次 → 放行（刹车生效）
    assert g._should_recommend_topic({"is_topic_clear": False, "topic_recommended": True}) == "outline"


# ==================== 2) 节点分支 / 兜底 ====================

async def test_analyze_topic_short_circuits_when_topic_given():
    g = _graph()
    out = await g._analyze_topic({"topic": "  强化学习  ", "keywords": []})
    assert out["is_topic_clear"] is True
    assert out["analysis_result"] == "选题明确：强化学习"


async def test_analyze_topic_calls_topic_agent_when_topic_empty():
    g = _graph()
    seen = {}

    async def fake_topic(payload):
        seen.update(payload)
        return {"topics": "1. 推荐选题A"}

    _patch_agents(g, topic=fake_topic)
    out = await g._analyze_topic({"topic": "", "keywords": ["RAG", "幻觉"]})
    assert out["is_topic_clear"] is False
    assert out["topic_recommended"] is True        # 刹车标记
    assert out["analysis_result"] == "1. 推荐选题A"
    assert seen["keywords"] == "RAG, 幻觉"


async def test_quality_check_fallback_on_error():
    """评审 Agent 抛异常 → 退回启发式评分，流程不崩"""
    g = _graph()

    async def boom(*args, **kwargs):
        raise RuntimeError("LLM 服务不可用")

    _patch_agents(g, review=boom)
    out = await g._quality_check({
        "final_paper": "正文存在", "references": [{"title": "R"}], "revision_count": 0,
    })
    assert out["quality_score"] == 0.8
    assert "基础评分" in out["quality_comment"]
    assert "LLM 服务不可用" in out["quality_comment"]
    assert out["quality_dimensions"] == []
    assert out["revision_count"] == 1


async def test_quality_check_fallback_when_parse_fails():
    """评审返回非法 JSON（data 无 score）→ 同样兜底；无正文无文献 = 0.5"""
    g = _graph()

    async def fake_review(*args, **kwargs):
        return {"raw": "不是 JSON", "data": {"raw": "不是 JSON"}}

    _patch_agents(g, review=fake_review)
    out = await g._quality_check({"final_paper": "", "references": [], "revision_count": 0})
    assert out["quality_score"] == 0.5


async def test_quality_check_clamps_out_of_range_score():
    g = _graph()

    async def fake_review(*args, **kwargs):
        return {"raw": "{}", "data": {"score": 1.7, "overall": "好", "dimensions": []}}

    _patch_agents(g, review=fake_review)
    assert (await g._quality_check({"revision_count": 0}))["quality_score"] == 1.0


async def test_quality_check_increments_revision_count():
    g = _graph()

    async def fake_review(*args, **kwargs):
        return {"raw": "{}", "data": {"score": 0.9, "overall": "好", "dimensions": []}}

    _patch_agents(g, review=fake_review)
    out = await g._quality_check({"revision_count": 4})
    assert out["revision_count"] == 5


async def test_write_sections_falls_back_when_outline_has_no_sections():
    """大纲里没有 sections（LLM 输出不稳定）→ 整篇当一节写"""
    g = _graph()
    seen = []

    async def fake_writer(payload):
        seen.append(payload)
        return {"content": "兜底正文"}

    _patch_agents(g, writer=fake_writer)
    out = await g._write_sections({"title": "标题X", "outline": {}, "references": []})
    assert list(out["sections_content"]) == ["标题X"]
    assert seen[0]["word_count"] == 800


async def test_add_references_falls_back_to_retrieval_when_state_empty():
    """state 里没有前置检索结果 → 退回「检索 + LLM 整理」兼容路径"""
    g = _graph()
    calls = {"retrieve": 0}

    async def fake_retrieve(query, top_k=5):
        calls["retrieve"] += 1
        return [{"id": "ref-001", "title": "论文A", "authors": "张三", "journal": "刊", "year": 2020}]

    async def fake_refs_execute(payload):
        return {"references": fake_refs_execute.docs, "formatted": "LLM 整理结果"}

    docs = [{"id": "ref-001", "title": "论文A", "authors": "张三", "journal": "刊", "year": 2020}]
    fake_refs_execute.docs = docs

    _patch_agents(g, retrieve=fake_retrieve, refs_execute=fake_refs_execute)
    out = await g._add_references({"title": "标题X", "references": []})
    assert out["formatted_references"] == "LLM 整理结果"
    assert out["references"] == docs


async def test_reference_retrieval_uses_title_then_topic():
    """检索 query 优先用标题；标题为空时退回选题"""
    g = _graph()
    queries = []

    async def fake_retrieve(query, top_k=5):
        queries.append(query)
        return []

    _patch_agents(g, retrieve=fake_retrieve)
    await g._retrieve_references({"title": "标题X", "topic": "方向Y"})
    await g._retrieve_references({"title": "", "topic": "方向Y"})
    assert queries == ["标题X", "方向Y"]


# ==================== 3) 整图运行 ====================

async def test_full_workflow_runs(monkeypatch):
    """全流程：大纲→检索→撰写→文献→润色→评审，最终状态带齐产出"""
    g = _graph()
    writer_inputs = []

    async def fake_topic(*a, **kw):
        return {"topics": "1. 推荐选题"}

    async def fake_outline(*a, **kw):
        return {"outline_dict": FAKE_OUTLINE}

    async def fake_retrieve(*a, **kw):
        return [{"id": "ref-001", "title": "论文A", "authors": "张三",
                 "journal": "NeurIPS", "year": 2020}]

    async def fake_writer(payload):
        writer_inputs.append(payload)
        return {"content": "章节正文内容。"}

    async def fake_polish(*a, **kw):
        return {"polished": "润色后的完整论文正文。"}

    async def fake_review(*a, **kw):
        return {"raw": "{}", "data": {"score": 0.9, "overall": "总体评价不错。",
                                      "dimensions": [{"dimension": "创新性与学术价值",
                                                      "score": 0.9, "comment": "有创新"}]}}

    _patch_agents(g, topic=fake_topic, outline=fake_outline, retrieve=fake_retrieve,
                  writer=fake_writer, polish=fake_polish, review=fake_review)

    state = await g.run(_base_state())

    assert state["final_paper"] == "润色后的完整论文正文。"
    assert state["quality_score"] == 0.9
    assert state["current_step"] == "quality_check"
    assert state["references"][0]["title"] == "论文A"
    # 确定性 GB/T 格式，且与正文引用编号同源同序
    assert state["references_text"] == "[1] 张三. 论文A[J]. NeurIPS. 2020."
    assert state["formatted_references"] == state["references_text"]
    assert set(state["sections_content"]) == {"引言", "方法"}
    assert state["revision_count"] == 1

    # 【回归断言】每章写作都必须拿到检索到的文献（修复前这里是空的）
    assert len(writer_inputs) == 2
    for payload in writer_inputs:
        assert payload["references"] == ["张三. 论文A[J]. NeurIPS. 2020."]


async def test_full_workflow_revises_on_low_score():
    """低分 → 打回重写：writer 收到 feedback，且会再次评审"""
    g = _graph()
    calls = {"writer": 0, "review": 0}

    async def fake_outline(*a, **kw):
        return {"outline_dict": FAKE_OUTLINE}

    async def fake_retrieve(*a, **kw):
        return [{"id": "ref-001", "title": "论文A", "authors": "张三",
                 "journal": "NeurIPS", "year": 2020}]

    async def fake_writer(payload):
        calls["writer"] += 1
        # 第二轮必须带上第一轮的评审意见
        if calls["writer"] > 2:
            assert "论证不足" in payload["feedback"]
        # 重写轮也必须继续带着文献（复用 state，不二次检索）
        assert payload["references"] == ["张三. 论文A[J]. NeurIPS. 2020."]
        return {"content": f"第{calls['writer']}轮正文。"}

    async def fake_polish(*a, **kw):
        return {"polished": "润色正文"}

    async def fake_review(*a, **kw):
        calls["review"] += 1
        if calls["review"] == 1:
            return {"raw": "{}", "data": {"score": 0.5, "overall": "论证不足，需改进", "dimensions": []}}
        return {"raw": "{}", "data": {"score": 0.9, "overall": "达标", "dimensions": []}}

    _patch_agents(g, topic=lambda *a, **kw: {"topics": ""}, outline=fake_outline,
                  retrieve=fake_retrieve, writer=fake_writer, polish=fake_polish, review=fake_review)

    state = await g.run(_base_state())

    assert calls["writer"] == 4       # 2 章 × 2 轮
    assert calls["review"] == 2       # 两轮评审
    assert state["quality_score"] == 0.9
    assert state["revision_count"] == 2


async def test_full_workflow_stops_at_max_revisions():
    """一直低分也不能死循环：到达 max_revisions 后放行"""
    g = _graph()
    calls = {"review": 0}

    async def fake_outline(*a, **kw):
        return {"outline_dict": {"sections": [{"title": "引言", "points": [], "word_count": 100}]}}

    async def fake_retrieve(*a, **kw):
        return []

    async def fake_writer(*a, **kw):
        return {"content": "正文"}

    async def fake_polish(*a, **kw):
        return {"polished": "全文"}

    async def fake_refs_execute(*a, **kw):
        return {"references": [], "formatted": ""}

    async def fake_review(*a, **kw):
        calls["review"] += 1
        return {"raw": "{}", "data": {"score": 0.1, "overall": "很差", "dimensions": []}}

    _patch_agents(g, outline=fake_outline, retrieve=fake_retrieve, writer=fake_writer,
                  refs_execute=fake_refs_execute, polish=fake_polish, review=fake_review)

    state = await g.run(_base_state(max_revisions=2))
    assert calls["review"] == 2                  # 首评 + 1 次重评
    assert state["revision_count"] == 2          # 到上限就停


async def test_workflow_stream_yields_node_events():
    """流式运行：能按节点收到事件"""
    g = _graph()

    async def fake_outline(*a, **kw):
        return {"outline_dict": {"sections": [{"title": "引言", "points": [], "word_count": 50}]}}

    async def fake_retrieve(*a, **kw):
        return []

    async def fake_writer(*a, **kw):
        return {"content": "正文"}

    async def fake_polish(*a, **kw):
        return {"polished": "全文"}

    async def fake_refs_execute(*a, **kw):
        return {"references": [], "formatted": ""}

    async def fake_review(*a, **kw):
        return {"raw": "{}", "data": {"score": 0.9, "overall": "好", "dimensions": []}}

    _patch_agents(g, outline=fake_outline, retrieve=fake_retrieve, writer=fake_writer,
                  refs_execute=fake_refs_execute, polish=fake_polish, review=fake_review)

    nodes = []
    async for event in g.stream(_base_state()):
        nodes.extend(event.keys())

    assert "quality_check" in nodes and "polish_paper" in nodes


# ==================== RAG 注入链路（本次修复的回归防线） ====================

async def test_retrieve_references_feeds_writer_prompt():
    """核心回归：检索节点排在写作之前，写作拿到编号化的文献列表"""
    g = _graph()
    seen = {}

    async def fake_retrieve(query, top_k=5):
        seen["query"] = query
        seen["top_k"] = top_k
        return [
            {"id": "ref-010", "title": "RAG 论文", "authors": "Lewis 等", "journal": "NeurIPS", "year": 2020},
            {"id": "ref-002", "title": "BERT", "authors": "Devlin 等", "journal": "NAACL", "year": 2019},
        ]

    async def fake_outline(*a, **kw):
        return {"outline_dict": {"sections": [{"title": "引言", "points": ["背景"], "word_count": 100}]}}

    async def fake_writer(payload):
        seen["refs"] = payload.get("references")
        return {"content": "正文"}

    async def fake_polish(*a, **kw):
        return {"polished": "全文"}

    async def fake_review(*a, **kw):
        return {"raw": "{}", "data": {"score": 0.9, "overall": "好", "dimensions": []}}

    _patch_agents(g, retrieve=fake_retrieve, outline=fake_outline, writer=fake_writer,
                  polish=fake_polish, review=fake_review)

    state = await g.run(_base_state())

    assert seen["query"] == "测试论文标题"
    assert seen["top_k"] == 5
    assert seen["refs"] == [
        "Lewis 等. RAG 论文[J]. NeurIPS. 2020.",
        "Devlin 等. BERT[J]. NAACL. 2019.",
    ]
    assert state["references_text"] == (
        "[1] Lewis 等. RAG 论文[J]. NeurIPS. 2020.\n"
        "[2] Devlin 等. BERT[J]. NAACL. 2019."
    )
    assert state["formatted_references"] == state["references_text"]


async def test_add_references_reuses_retrieval_without_extra_llm_call():
    """检索只发生一次：文献整理节点复用前置结果，不重复检索、不再发 LLM 请求"""
    g = _graph()
    counters = {"retrieve": 0, "llm_format": 0}

    async def fake_retrieve(*a, **kw):
        counters["retrieve"] += 1
        return [{"id": "ref-001", "title": "论文A", "authors": "张三", "journal": "NeurIPS", "year": 2020}]

    async def fake_refs_execute(*a, **kw):
        counters["llm_format"] += 1
        return {"references": [], "formatted": "不该被调用"}

    async def fake_outline(*a, **kw):
        return {"outline_dict": {"sections": [{"title": "引言", "points": [], "word_count": 100}]}}

    async def fake_writer(*a, **kw):
        return {"content": "正文"}

    async def fake_polish(*a, **kw):
        return {"polished": "全文"}

    async def fake_review(*a, **kw):
        return {"raw": "{}", "data": {"score": 0.9, "overall": "好", "dimensions": []}}

    _patch_agents(g, retrieve=fake_retrieve, refs_execute=fake_refs_execute, outline=fake_outline,
                  writer=fake_writer, polish=fake_polish, review=fake_review)

    state = await g.run(_base_state())

    assert counters["retrieve"] == 1        # 一轮流程只检索一次
    assert counters["llm_format"] == 0      # 不再多花一次 LLM 调用
    assert state["formatted_references"] == "[1] 张三. 论文A[J]. NeurIPS. 2020."


def test_run_uses_explicit_thread_id_when_config_given():
    """传了 config 就用它；没传则按 paper_id 生成独立线程"""
    from app.workflows.graph_builder import PaperWorkflowGraph as G

    assert G._thread_config(None, {"paper_id": 7}) == {"configurable": {"thread_id": "paper-7"}}
    assert G._thread_config(None, {"paper_id": None}) == {"configurable": {"thread_id": "paper-anon"}}
    cfg = {"configurable": {"thread_id": "custom"}}
    assert G._thread_config(cfg, {"paper_id": 7}) is cfg

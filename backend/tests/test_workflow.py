# 单元测试：工作流图的路由逻辑 + 用假 Agent 跑全图（不联网、不花 token）
from app.workflows.graph_builder import PaperWorkflowGraph

FAKE_OUTLINE = {"sections": [
    {"title": "引言", "points": ["背景介绍"], "word_count": 200},
    {"title": "方法", "points": ["技术方案"], "word_count": 300},
]}


def _graph():
    return PaperWorkflowGraph()


# ---------- 路由函数（纯逻辑，不调用 Agent） ----------

def test_check_quality_pass_on_high_score():
    g = _graph()
    assert g._check_quality({"quality_score": 0.7}) == "pass"
    assert g._check_quality({"quality_score": 0.95}) == "pass"


def test_check_quality_revise_then_exhausted():
    g = _graph()
    # 低分且次数未用尽 → 打回重写
    assert g._check_quality({"quality_score": 0.5, "revision_count": 0, "max_revisions": 2}) == "revise"
    # 低分但次数已用尽 → 放行结束（防死循环）
    assert g._check_quality({"quality_score": 0.5, "revision_count": 2, "max_revisions": 2}) == "pass"


def test_should_recommend_topic_routing():
    g = _graph()
    assert g._should_recommend_topic({"is_topic_clear": True}) == "outline"
    assert g._should_recommend_topic({"is_topic_clear": False}) == "recommend"
    # 已推荐过一次 → 放行（刹车生效，防死循环）
    assert g._should_recommend_topic({"is_topic_clear": False, "topic_recommended": True}) == "outline"


# ---------- _quality_check 的兜底逻辑 ----------

async def test_quality_check_fallback_on_error(monkeypatch):
    """评审 Agent 抛异常时退回启发式评分，流程不崩"""
    g = _graph()

    async def boom(**kwargs):
        raise RuntimeError("LLM 服务不可用")

    monkeypatch.setattr(g.reviewer_agent, "execute", boom)
    out = await g._quality_check({
        "final_paper": "正文存在", "references": [{"title": "R"}], "revision_count": 0,
    })
    assert out["quality_score"] == 0.8          # 有正文+有文献 → 0.8
    assert "基础评分" in out["quality_comment"]
    assert out["quality_dimensions"] == []


async def test_quality_check_fallback_when_parse_fails(monkeypatch):
    """评审返回非法 JSON（data 为空）时同样兜底"""
    g = _graph()

    async def fake_review(**kwargs):
        return {"raw": "不是 JSON", "data": {"raw": "不是 JSON"}}

    monkeypatch.setattr(g.reviewer_agent, "execute", fake_review)
    out = await g._quality_check({"final_paper": "", "references": [], "revision_count": 0})
    assert out["quality_score"] == 0.5          # 无正文无文献 → 0.5


# ---------- 完整图跑一遍（假 Agent 全替换） ----------

async def test_full_workflow_runs(monkeypatch):
    """6 节点全流程：大纲→撰写→文献→润色→评审，最终状态带齐产出"""
    g = _graph()

    async def fake_topic(*args, **kwargs):
        return {"topics": "1. 推荐选题"}

    async def fake_outline(*args, **kwargs):
        return {"outline_dict": FAKE_OUTLINE}

    async def fake_writer(*args, **kwargs):
        return {"content": "章节正文内容。"}

    async def fake_refs(*args, **kwargs):
        return {
            "references": [{"title": "论文A", "authors": "张三", "year": 2020}],
            "formatted": "1. 张三. 论文A（2020）",
        }

    async def fake_polish(*args, **kwargs):
        return {"polished": "润色后的完整论文正文。"}

    async def fake_review(*args, **kwargs):
        return {
            "raw": "{}",
            "data": {
                "score": 0.9,
                "overall": "总体评价不错。",
                "dimensions": [{"dimension": "创新性与学术价值", "score": 0.9, "comment": "有创新"}],
            },
        }

    monkeypatch.setattr(g.topic_agent, "execute", fake_topic)
    monkeypatch.setattr(g.outline_agent, "execute", fake_outline)
    monkeypatch.setattr(g.writer_agent, "execute", fake_writer)
    monkeypatch.setattr(g.reference_agent, "execute", fake_refs)
    monkeypatch.setattr(g.polish_agent, "execute", fake_polish)
    monkeypatch.setattr(g.reviewer_agent, "execute", fake_review)

    state = await g.run({
        "paper_id": 1,
        "title": "测试论文标题",
        "topic": "人工智能",
        "keywords": ["AI"],
        "is_topic_clear": True,
        "revision_count": 0,
        "max_revisions": 2,
        "sections_content": {},
    })

    assert state["final_paper"] == "润色后的完整论文正文。"
    assert state["quality_score"] == 0.9
    assert state["current_step"] == "quality_check"
    assert state["references"][0]["title"] == "论文A"
    assert state["formatted_references"] == "1. 张三. 论文A（2020）"
    assert set(state["sections_content"]) == {"引言", "方法"}
    assert state["revision_count"] == 1          # 评审节点执行了一次


async def test_full_workflow_revises_on_low_score(monkeypatch):
    """低分论文触发打回重写：writer 会收到 feedback，并再次评审"""
    g = _graph()
    calls = {"writer": 0, "review": 0}

    async def fake_outline(*args, **kwargs):
        return {"outline_dict": FAKE_OUTLINE}

    async def fake_writer(*args, **kwargs):
        calls["writer"] += 1
        data = args[0] if args else {}
        assert data.get("feedback", "") == "" or "改进" in data.get("feedback", "")
        return {"content": f"第{calls['writer']}轮正文。"}

    async def fake_refs(*args, **kwargs):
        return {"references": [{"title": "论文A", "authors": "张三", "year": 2020}], "formatted": "1. 张三. 论文A（2020）"}

    async def fake_polish(*args, **kwargs):
        return {"polished": "润色正文"}

    async def fake_review(*args, **kwargs):
        calls["review"] += 1
        if calls["review"] == 1:
            return {"raw": "{}", "data": {"score": 0.5, "overall": "需要改进：论证不足", "dimensions": []}}
        return {"raw": "{}", "data": {"score": 0.9, "overall": "达标", "dimensions": []}}

    monkeypatch.setattr(g.topic_agent, "execute", lambda *a, **kw: {"topics": ""})
    monkeypatch.setattr(g.outline_agent, "execute", fake_outline)
    monkeypatch.setattr(g.writer_agent, "execute", fake_writer)
    monkeypatch.setattr(g.reference_agent, "execute", fake_refs)
    monkeypatch.setattr(g.polish_agent, "execute", fake_polish)
    monkeypatch.setattr(g.reviewer_agent, "execute", fake_review)

    state = await g.run({
        "paper_id": 1, "title": "测试论文", "topic": "AI",
        "keywords": [], "is_topic_clear": True,
        "revision_count": 0, "max_revisions": 2, "sections_content": {},
    })

    assert calls["writer"] == 4       # 2 个章节 × 2 轮（首写 + 重写）
    assert calls["review"] == 2       # 两轮评审
    assert state["quality_score"] == 0.9

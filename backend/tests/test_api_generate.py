# 集成测试：生成接口（假工作流）+ 质量评审持久化 + SSE 事件流
import json

from app.api import papers as papers_api
from app.config import settings

FAKE_STATE = {
    "final_paper": "## 引言\n\n这是一段正文内容。",
    "outline": {"sections": [{"title": "引言", "points": ["背景介绍"], "word_count": 200}]},
    "sections_content": {"引言": "这是一段正文内容。"},
    "references": [{"title": "论文A", "authors": "张三", "journal": "期刊X", "year": 2020}],
    "formatted_references": "[1] 张三. 论文A[J]. 期刊X. 2020.",
    "quality_score": 0.85,
    "quality_comment": "总体评价：结构完整，建议补充对比实验。",
    "quality_dimensions": [
        {"dimension": "创新性与学术价值", "score": 0.8, "comment": "有一定创新"},
        {"dimension": "逻辑结构与论证", "score": 0.9, "comment": "论证清晰"},
    ],
}


async def _create(client, headers, title="生成测试论文"):
    r = await client.post("/api/papers/", json={"title": title, "topic": "人工智能"}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def _patch_run(monkeypatch, state=None):
    async def fake_run(initial_state):
        return dict(state if state is not None else FAKE_STATE)

    monkeypatch.setattr(papers_api.paper_workflow, "run", fake_run)


# ---------- 同步生成 + 持久化 ----------

async def test_generate_saves_quality(client, headers, monkeypatch):
    """核心回归：生成后 quality 三字段落库，详情接口能读回（评审卡片刷新不消失）"""
    _patch_run(monkeypatch)

    p = await _create(client, headers)
    r = await client.post(f"/api/papers/{p['id']}/generate", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["quality_score"] == 0.85
    assert body["sections"] == 1
    assert body["references"] == 1

    d = (await client.get(f"/api/papers/{p['id']}", headers=headers)).json()
    assert d["content"] == FAKE_STATE["final_paper"]
    assert d["status"] == "completed"
    assert d["quality_score"] == 0.85
    assert d["quality_comment"] == FAKE_STATE["quality_comment"]
    dims = json.loads(d["quality_dimensions"])
    assert len(dims) == 2
    assert dims[0]["dimension"] == "创新性与学术价值"
    assert len(d["chapters"]) == 1
    assert len(d["references"]) == 1


async def test_generate_saves_outline_and_chapters(client, headers, monkeypatch):
    _patch_run(monkeypatch)
    p = await _create(client, headers)
    await client.post(f"/api/papers/{p['id']}/generate", headers=headers)
    d = (await client.get(f"/api/papers/{p['id']}", headers=headers)).json()
    assert json.loads(d["outline"])["sections"][0]["title"] == "引言"
    assert d["chapters"][0]["title"] == "引言"
    assert d["chapters"][0]["order_index"] == 0


async def test_generate_dedupes_repeated_reference_titles(client, headers, monkeypatch):
    """同一标题的文献只落一条，避免重复灌库时参考文献翻倍"""
    state = dict(FAKE_STATE)
    state["references"] = [
        {"title": "论文A", "authors": "张三", "year": 2020},
        {"title": "论文A", "authors": "张三", "year": 2020},   # 重复
        {"title": "论文B", "authors": "李四", "year": 2021},
    ]
    _patch_run(monkeypatch, state)

    p = await _create(client, headers)
    r = await client.post(f"/api/papers/{p['id']}/generate", headers=headers)
    assert r.status_code == 200
    d = (await client.get(f"/api/papers/{p['id']}", headers=headers)).json()
    assert sorted(ref["title"] for ref in d["references"]) == ["论文A", "论文B"]


async def test_generate_skips_reference_without_title(client, headers, monkeypatch):
    """缺标题的文献直接跳过，不进库"""
    state = dict(FAKE_STATE)
    state["references"] = [{"authors": "无名", "year": 2020}, {"title": "论文B", "year": 2021}]
    _patch_run(monkeypatch, state)

    p = await _create(client, headers)
    await client.post(f"/api/papers/{p['id']}/generate", headers=headers)
    d = (await client.get(f"/api/papers/{p['id']}", headers=headers)).json()
    assert [ref["title"] for ref in d["references"]] == ["论文B"]


async def test_generate_without_content_stays_draft(client, headers, monkeypatch):
    """工作流没产出正文 → 状态保持 draft"""
    _patch_run(monkeypatch, {"final_paper": "", "sections_content": {}, "references": []})

    p = await _create(client, headers)
    await client.post(f"/api/papers/{p['id']}/generate", headers=headers)
    d = (await client.get(f"/api/papers/{p['id']}", headers=headers)).json()
    assert d["status"] == "draft"


async def test_generate_regenerates_clears_old_chapters(client, headers, monkeypatch):
    """二次生成：旧章节要被替换，而不是累加"""
    _patch_run(monkeypatch)
    p = await _create(client, headers)
    await client.post(f"/api/papers/{p['id']}/generate", headers=headers)
    await client.post(f"/api/papers/{p['id']}/generate", headers=headers)
    d = (await client.get(f"/api/papers/{p['id']}", headers=headers)).json()
    assert len(d["chapters"]) == 1
    assert len(d["references"]) == 1


async def test_generate_requires_auth(client):
    assert (await client.post("/api/papers/1/generate")).status_code == 401


async def test_generate_paper_not_found(client, headers, monkeypatch):
    _patch_run(monkeypatch)
    assert (await client.post("/api/papers/9999/generate", headers=headers)).status_code == 404


async def test_generate_cross_user_404(client, user_pair, monkeypatch):
    _patch_run(monkeypatch)
    h1, h2 = user_pair
    p = await _create(client, h1)
    assert (await client.post(f"/api/papers/{p['id']}/generate", headers=h2)).status_code == 404


async def test_generate_fails_when_keys_missing(client, headers, monkeypatch):
    """没配 Key 时给出 400 与可读提示，而不是 500"""
    _patch_run(monkeypatch)
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", None)
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", None)
    p = await _create(client, headers)
    r = await client.post(f"/api/papers/{p['id']}/generate", headers=headers)
    assert r.status_code == 400
    assert "DEEPSEEK_API_KEY" in r.json()["detail"]


# ---------- 流式生成（SSE） ----------

def _patch_stream(monkeypatch, events):
    async def fake_stream(initial_state):
        for e in events:
            yield e

    monkeypatch.setattr(papers_api.paper_workflow, "stream", fake_stream)


async def test_stream_generate_done_event_with_quality(client, headers, monkeypatch):
    """SSE done 事件必须携带 quality_score（前端卡片数据源）"""
    _patch_stream(monkeypatch, [
        {"generate_outline": {"current_step": "outline", "outline": FAKE_STATE["outline"]}},
        {"quality_check": {
            "current_step": "quality_check",
            "quality_score": 0.85,
            "quality_comment": FAKE_STATE["quality_comment"],
            "quality_dimensions": FAKE_STATE["quality_dimensions"],
        }},
    ])

    p = await _create(client, headers)
    async with client.stream("POST", f"/api/papers/{p['id']}/stream-generate", headers=headers) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = [ln async for ln in resp.aiter_lines()]

    events = [json.loads(ln[6:]) for ln in lines if ln.startswith("data: ")]
    assert len(events) == 3                       # 2 个节点事件 + 1 个 done
    done = [e for e in events if e.get("done")]
    assert done, "SSE 流里必须有 done 事件"
    assert done[0]["quality_score"] == 0.85
    assert done[0]["quality_comment"] == FAKE_STATE["quality_comment"]
    assert len(done[0]["quality_dimensions"]) == 2


async def test_stream_generate_persists_after_done(client, headers, monkeypatch):
    """流式结束后数据同样落库"""
    _patch_stream(monkeypatch, [
        {"polish_paper": {"final_paper": "流式产出的全文。"}},
        {"quality_check": {
            "current_step": "quality_check",
            "quality_score": 0.85,
            "quality_comment": "流式评审意见",
            "quality_dimensions": [],
        }},
    ])

    p = await _create(client, headers)
    async with client.stream("POST", f"/api/papers/{p['id']}/stream-generate", headers=headers) as resp:
        await resp.aread()

    d = (await client.get(f"/api/papers/{p['id']}", headers=headers)).json()
    assert d["quality_score"] == 0.85
    assert d["quality_comment"] == "流式评审意见"
    assert d["content"] == "流式产出的全文。"
    assert d["status"] == "completed"


async def test_stream_generate_error_event(client, headers, monkeypatch):
    """工作流异常时 SSE 输出 error 事件而非崩溃"""
    async def fake_stream(initial_state):
        raise RuntimeError("模拟生成失败")
        yield  # pragma: no cover

    monkeypatch.setattr(papers_api.paper_workflow, "stream", fake_stream)

    p = await _create(client, headers)
    async with client.stream("POST", f"/api/papers/{p['id']}/stream-generate", headers=headers) as resp:
        assert resp.status_code == 200
        lines = [ln async for ln in resp.aiter_lines()]

    events = [json.loads(ln[6:]) for ln in lines if ln.startswith("data: ")]
    errors = [e for e in events if e.get("error")]
    assert errors and "模拟生成失败" in errors[0]["error"]


async def test_stream_generate_not_found(client, headers):
    async with client.stream("POST", "/api/papers/9999/stream-generate", headers=headers) as resp:
        assert resp.status_code == 404


async def test_stream_generate_requires_auth(client):
    async with client.stream("POST", "/api/papers/1/stream-generate") as resp:
        assert resp.status_code == 401

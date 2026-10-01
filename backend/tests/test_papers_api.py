# 集成测试：论文 CRUD + 搜索 / 筛选 / 分页 / 权限
import pytest


async def _create(client, headers, title="这是一篇测试论文", **kw):
    payload = {"title": title, **kw}
    r = await client.post("/api/papers/", json=payload, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


# ---------- 创建 ----------

async def test_create_requires_auth(client):
    r = await client.post("/api/papers/", json={"title": "未登录创建"})
    assert r.status_code == 401


async def test_create_ok(client, headers):
    p = await _create(client, headers, topic="人工智能", keywords="AI,论文")
    assert p["title"] == "这是一篇测试论文"
    assert p["topic"] == "人工智能"
    assert p["status"] == "draft"


async def test_create_title_too_short(client, headers):
    r = await client.post("/api/papers/", json={"title": "短"}, headers=headers)
    assert r.status_code == 422          # Pydantic 校验：标题最少 5 字


# ---------- 列表 / 搜索 / 筛选 / 分页 ----------

async def test_list_pagination(client, headers):
    for i in range(3):
        await _create(client, headers, title=f"分页论文第{i}篇")
    r = await client.get("/api/papers/", params={"skip": 0, "limit": 2}, headers=headers)
    body = r.json()
    assert r.status_code == 200
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["skip"] == 0 and body["limit"] == 2


async def test_list_search_by_keyword(client, headers):
    await _create(client, headers, title="关于大语言模型的研究")
    await _create(client, headers, title="关于推荐系统的研究")
    r = await client.get("/api/papers/", params={"q": "大语言模型"}, headers=headers)
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["title"] == "关于大语言模型的研究"


async def test_list_search_by_topic(client, headers):
    await _create(client, headers, title="论文甲：自然语言处理方向", topic="自然语言处理")
    await _create(client, headers, title="论文乙：计算机视觉方向", topic="计算机视觉")
    r = await client.get("/api/papers/", params={"q": "计算机视觉"}, headers=headers)
    assert r.json()["total"] == 1


async def test_list_filter_status(client, headers):
    p = await _create(client, headers)
    await _create(client, headers)
    # 把第一篇改成已完成
    r = await client.put(f"/api/papers/{p['id']}", json={"status": "completed"}, headers=headers)
    assert r.status_code == 200
    r = await client.get("/api/papers/", params={"status_filter": "completed"}, headers=headers)
    assert r.json()["total"] == 1
    r = await client.get("/api/papers/", params={"status_filter": "draft"}, headers=headers)
    assert r.json()["total"] == 1


async def test_list_isolation_between_users(client, user_pair):
    """用户只能看到自己的论文"""
    h1, h2 = user_pair
    await _create(client, h1, title="属于用户甲的论文")
    r = await client.get("/api/papers/", headers=h2)
    assert r.json()["total"] == 0


# ---------- 详情 / 更新 / 删除 ----------

async def test_detail_contains_quality_fields(client, headers):
    """详情接口返回质量评审字段（生成前为 None）"""
    p = await _create(client, headers)
    r = await client.get(f"/api/papers/{p['id']}", headers=headers)
    body = r.json()
    assert body["id"] == p["id"]
    assert body["quality_score"] is None
    assert body["quality_comment"] is None
    assert body["quality_dimensions"] is None


async def test_detail_not_found(client, headers):
    assert (await client.get("/api/papers/9999", headers=headers)).status_code == 404


async def test_update_partial(client, headers):
    p = await _create(client, headers, abstract="旧摘要")
    r = await client.put(f"/api/papers/{p['id']}", json={"abstract": "新摘要"}, headers=headers)
    assert r.status_code == 200
    assert r.json()["abstract"] == "新摘要"
    assert r.json()["title"] == "这是一篇测试论文"    # 未传字段保持原值


async def test_delete_and_verify(client, headers):
    p = await _create(client, headers)
    r = await client.delete(f"/api/papers/{p['id']}", headers=headers)
    assert r.status_code == 204
    assert (await client.get(f"/api/papers/{p['id']}", headers=headers)).status_code == 404


# ---------- 越权（只能操作自己的论文） ----------

async def test_cross_user_detail_404(client, user_pair):
    h1, h2 = user_pair
    p = await _create(client, h1)
    r = await client.get(f"/api/papers/{p['id']}", headers=h2)
    assert r.status_code == 404


async def test_cross_user_update_404(client, user_pair):
    h1, h2 = user_pair
    p = await _create(client, h1)
    r = await client.put(f"/api/papers/{p['id']}", json={"abstract": "x"}, headers=h2)
    assert r.status_code == 404


async def test_cross_user_delete_404(client, user_pair):
    h1, h2 = user_pair
    p = await _create(client, h1)
    r = await client.delete(f"/api/papers/{p['id']}", headers=h2)
    assert r.status_code == 404
    # 论文还在（未被删）
    assert (await client.get(f"/api/papers/{p['id']}", headers=h1)).status_code == 200

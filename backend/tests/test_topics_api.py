# 集成测试：选题推荐接口（假 TopicAgent）
from app.api import topics as topics_api


async def test_recommend_ok(client, headers, monkeypatch):
    async def fake_execute(*args, **kwargs):
        return {"topics": "1. 基于大模型的文本摘要研究\n2. 多模态检索增强生成"}

    monkeypatch.setattr(topics_api.topic_agent, "execute", fake_execute)

    r = await client.post("/api/topics/recommend", json={"field": "人工智能"}, headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert "文本摘要" in body["topics"]
    assert "多模态" in body["topics"]


async def test_recommend_empty_field(client, headers):
    r = await client.post("/api/topics/recommend", json={"field": "   "}, headers=headers)
    assert r.status_code == 400


async def test_recommend_requires_auth(client):
    r = await client.post("/api/topics/recommend", json={"field": "AI"})
    assert r.status_code == 401

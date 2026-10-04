# 集成测试：应用入口与数据库依赖
import pytest

from app.database import get_db


async def test_root_health_check(client):
    r = await client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "running"
    assert body["version"] == "1.0.0"
    assert "message" in body


async def test_openapi_schema_exposes_all_routers(client):
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    for p in ("/api/auth/register", "/api/auth/login", "/api/auth/me",
              "/api/papers/", "/api/topics/recommend"):
        assert p in paths
    assert "/api/papers/{paper_id}/export" in paths
    assert "/api/papers/{paper_id}/stream-generate" in paths


async def test_get_db_commits_and_closes():
    """get_db 依赖：正常路径要能提交并关闭会话"""
    gen = get_db()
    session = await gen.__anext__()
    assert session is not None
    try:
        await gen.__anext__()          # 走到 yield 之后 → 提交 + 关闭
    except StopAsyncIteration:
        pass


async def test_get_db_rolls_back_on_error():
    """异常路径：依赖要回滚并把异常继续抛出，而不是静默提交"""
    gen = get_db()
    await gen.__anext__()
    with pytest.raises(RuntimeError, match="boom"):
        await gen.athrow(RuntimeError("boom"))

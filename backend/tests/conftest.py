# conftest.py —— 测试环境公共配置
# 必须在 import app 之前覆盖环境变量：pydantic-settings 环境变量优先于 .env，
# 这样测试跑在独立库上，绝不碰开发库 data/paper_writer.db。
import os

os.environ["DATABASE_URL"] = "sqlite:///./data/test_paper_writer.db"
os.environ["VECTOR_DB_PATH"] = "./data/test_chroma"

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from helpers import register_and_login  # noqa: F401  供各测试文件复用

# 注意顺序：`import app.models.*` 会把名字 `app` 重新绑定为「包模块」，
# 若放在 `from app.main import app` 之后，app 会被覆盖成模块导致 ASGI 调用失败，
# 所以模型导入必须在前，确保最后拿到的 app 是 FastAPI 实例。
import app.models.user  # noqa: F401  确保模型注册进 Base.metadata
import app.models.paper  # noqa: F401
import app.models.chapter  # noqa: F401
import app.models.reference  # noqa: F401

from app.database import AsyncSessionLocal, Base, engine
from app.main import _ensure_quality_columns, app


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def init_db():
    """会话级：建表 + 补列迁移（等价于后端启动时的 lifespan 动作）"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_quality_columns(conn)
    yield
    await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def clean_tables(init_db):
    """每个用例前清空所有表，保证用例间互不影响"""
    async with AsyncSessionLocal() as db:
        for t in ("chapters", "references", "papers", "users"):
            await db.execute(text(f'DELETE FROM "{t}"'))
        await db.commit()
    yield


@pytest_asyncio.fixture
async def client():
    """FastAPI 测试客户端（ASGI 直连，不启真实端口；不跑 lifespan）"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def token(client):
    return await register_and_login(client)


@pytest_asyncio.fixture
async def headers(token):
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def user_pair(client):
    """两个不同用户，用于越权测试"""
    t1 = await register_and_login(client, "user_a")
    t2 = await register_and_login(client, "user_b")
    return {"Authorization": f"Bearer {t1}"}, {"Authorization": f"Bearer {t2}"}

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from app.config import settings

database_url = settings.DATABASE_URL
if database_url.startswith("sqlite"):
    database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://", 1)

engine = create_async_engine(database_url, echo=settings.DEBUG, future=True)

# 会话工厂：必须先调它才拿到 Session。
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# ORM 模型基类：所有表模型继承它，元数据才被收集。
Base = declarative_base()

# FastAPI 依赖函数：每个请求执行一次，yield 前是 setup、yield 后是 teardown
async def get_db():
    async with AsyncSessionLocal() as session:#退出后自动关闭会话
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
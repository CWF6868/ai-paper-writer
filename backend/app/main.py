# main.py —— FastAPI 应用入口
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from sqlalchemy import text

from app.database import engine, Base
# 导入四个模型，create_all 才能认出所有表
import app.models.user, app.models.paper, app.models.chapter, app.models.reference
from app.api import papers, auth, topics, export

async def _ensure_quality_columns(conn):
    """轻量迁移：给已存在的 papers 表补质量评审三列（create_all 不改已有表结构）"""
    res = await conn.execute(text("PRAGMA table_info(papers)"))
    existing = {row[1] for row in res.fetchall()}
    additions = {
        "quality_score": "FLOAT",
        "quality_comment": "TEXT",
        "quality_dimensions": "TEXT",
    }
    for name, typ in additions.items():
        if name not in existing:
            await conn.execute(text(f"ALTER TABLE papers ADD COLUMN {name} {typ}"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时建表、补列迁移、初始化向量库；关闭时释放连接"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_quality_columns(conn)
    from app.rag.vector_store import vector_store  # noqa: F401  碰一下单例即完成初始化
    yield
    await engine.dispose()

app = FastAPI(
    title="AI 论文写作系统",
    description="基于多 Agent 协作的智能论文写作平台",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS：允许前端开发服务器跨域访问（第八步前端联调要用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由（auth/chapters/references 会在后续步骤补上）
app.include_router(papers.router, prefix="/api/papers", tags=["论文"])
app.include_router(export.router, prefix="/api/papers", tags=["导出"])
app.include_router(topics.router, prefix="/api/topics", tags=["选题"])
app.include_router(auth.router, prefix= "/api/auth" , tags=[ "认证" ]) # ← 新增

@app.get("/")
async def root():
    """健康检查"""
    return {"message": "AI 论文写作系统 API", "version": "1.0.0", "status": "running"}
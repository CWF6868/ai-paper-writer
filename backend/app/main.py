# main.py —— FastAPI 应用入口
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from sqlalchemy import text

from app.config import settings
from app.database import engine, Base
# 导入模型，create_all 才能认出所有表
import app.models.user, app.models.paper, app.models.chapter, app.models.reference
import app.models.llm_call_log  # noqa: F401  P0-3 成本流水表
from app.api import papers, auth, topics, export, metrics
from app.core.llm_log_sink import usage_log_sink

logger = logging.getLogger(__name__)

async def _ensure_quality_columns(conn):
    """轻量迁移：给已存在的 papers 表补质量评审列（create_all 不改已有表结构）"""
    res = await conn.execute(text("PRAGMA table_info(papers)"))
    existing = {row[1] for row in res.fetchall()}
    additions = {
        "quality_score": "FLOAT",
        "quality_comment": "TEXT",
        "quality_dimensions": "TEXT",
        # 老库补列必须给 DEFAULT，否则 ADD COLUMN NOT NULL 会因存量行无值而失败
        "quality_degraded": "BOOLEAN NOT NULL DEFAULT 0",
    }
    for name, typ in additions.items():
        if name not in existing:
            await conn.execute(text(f"ALTER TABLE papers ADD COLUMN {name} {typ}"))


async def _ensure_llm_log_columns(conn):
    """轻量迁移：给已存在的 llm_call_log 表补 user_id。

    create_all 只建新表、不改已有表结构。P0-3 第 2 步建表时还没有 user_id，
    老库升级后如果不补这一列，落库会直接失败（no such column），
    而 sink 的失败策略是「丢弃 + 计数」，于是成本表会静默停止增长。
    """
    res = await conn.execute(text("PRAGMA table_info(llm_call_log)"))
    existing = {row[1] for row in res.fetchall()}
    if not existing:            # 表还不存在（全新库），create_all 会建好
        return
    if "user_id" not in existing:
        await conn.execute(text("ALTER TABLE llm_call_log ADD COLUMN user_id INTEGER"))
    await conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_llm_call_log_user_created "
        "ON llm_call_log (user_id, created_at)"
    ))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时建表、补列迁移、初始化向量库；关闭时释放连接"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_quality_columns(conn)
        await _ensure_llm_log_columns(conn)
    from app.rag.vector_store import vector_store  # noqa: F401  碰一下单例即完成初始化
    # 启动成本流水落库（绑定事件循环）。不启动的话，回调只能丢弃记录。
    usage_log_sink.start()

    # 演示模式：首次启动准备语料 / 演示账号 / 一篇成稿。
    # 失败不阻断启动：Demo 少一篇样例不等于服务不可用，日志留证据即可。
    if settings.DEMO_MODE and settings.DEMO_SEED_ON_START:
        from app.demo.seed import ensure_demo_data
        try:
            stats = await ensure_demo_data()
            logger.info("演示数据就绪：%s", stats)
        except Exception:  # noqa: BLE001
            logger.exception("演示数据初始化失败（服务继续启动，Demo 可能不完整）")

    yield
    # 退出前把队列里还没落库的调用流水刷盘：异步落库是批量的，
    # 不在这里收尾，最后一批（最多 50 条）会随进程一起消失。
    await usage_log_sink.stop()
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
app.include_router(metrics.router, prefix="/api/metrics", tags=["成本指标"])
app.include_router(auth.router, prefix= "/api/auth" , tags=[ "认证" ]) # ← 新增

@app.get("/")
async def root():
    """健康检查"""
    return {"message": "AI 论文写作系统 API", "version": "1.0.0", "status": "running"}
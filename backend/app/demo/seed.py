# demo/seed.py —— 演示数据的首次初始化（幂等）
#
# 三件事，缺一则 Demo 看起来像坏了一半：
#   1) 向量库要有 42 篇语料 —— data/ 在 .gitignore 里，镜像里天然是空库，
#      不灌库的话 RAG 会静默失效（正文没有引用，但页面不会报错）；
#   2) 要有一个可登录的演示账号 —— 面试官不该先花 30 秒注册；
#   3) 要有一篇已生成的成稿 —— 首屏空列表等于让人以为项目没跑起来。
#
# 幂等：重复启动不会重复建用户、不会重复生成论文、不会重复计费（演示模式本来也不计费）。
import logging

from sqlalchemy import select

from app.config import settings
from app.core.llm_client import build_embedding_model
from app.core.security import hash_password
from app.database import AsyncSessionLocal
from app.models.paper import Paper, PaperStatus
from app.models.user import User
from app.rag.corpus import ACADEMIC_REFERENCES

logger = logging.getLogger(__name__)

DEMO_USERNAME = "demo"
DEMO_EMAIL = "demo@example.com"
DEMO_PASSWORD = "demo-readonly"
DEMO_PAPER_TITLE = "检索增强生成在学术写作中的应用研究"
DEMO_PAPER_TOPIC = "大语言模型与检索增强生成"
DEMO_PAPER_KEYWORDS = "RAG,大语言模型,学术写作,文献综述"


def configure_demo_embeddings() -> int:
    """把向量库的 embedding 换成本地实现，并保证语料已入库。返回库内文献数。

    为什么必须显式覆盖：vector_store 是模块级单例，如果它在本进程里已经被
    真实 embedding 初始化过（测试、或先连了真实 Key 再切 demo），
    继续用它灌库会真的发起网络请求 —— 演示模式"零外呼"的承诺就破了。

    集合也要切：正式库是 1024 维，演示是 256 维，同一集合里混用 Chroma 会报
    "Embedding dimension does not match collection dimensionality"。
    """
    from app.demo.chat import DemoHashEmbeddings
    from app.rag.vector_store import DEMO_COLLECTION_NAME, vector_store

    vector_store.use_collection(DEMO_COLLECTION_NAME)
    if not isinstance(vector_store.embeddings, DemoHashEmbeddings):
        vector_store.embeddings = build_embedding_model()

    count = vector_store.get_document_count()
    if count < len(ACADEMIC_REFERENCES):
        vector_store.add_documents(ACADEMIC_REFERENCES)
        count = vector_store.get_document_count()
        logger.info("演示语料已入库：%d 篇", count)
    return count


async def _ensure_user(db) -> User:
    user = (await db.execute(
        select(User).where(User.email == DEMO_EMAIL)
    )).scalar_one_or_none()
    if user:
        return user

    user = User(
        username=DEMO_USERNAME,
        email=DEMO_EMAIL,
        hashed_password=hash_password(DEMO_PASSWORD),
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info("已创建演示账号：%s / %s", DEMO_EMAIL, DEMO_PASSWORD)
    return user


async def _ensure_sample_paper(db, user: User) -> Paper:
    """演示论文：真的跑一遍工作流再落库，而不是塞一段静态文本。

    走真实流程的好处是：章节、文献、质量分、字数统计都由代码算出来，
    演示页上的数字与真实生成完全同源，经得起"这是不是写死的"追问。
    """
    paper = (await db.execute(
        select(Paper).where(
            Paper.author_id == user.id,
            Paper.title == DEMO_PAPER_TITLE,
        )
    )).scalar_one_or_none()
    # 有记录但正文为空 = 上次初始化中途失败（比如向量库维度不匹配）。
    # 这种半成品必须重跑，否则演示首屏永远是一篇空稿。
    if paper and (paper.content or "").strip():
        return paper

    if paper is None:
        paper = Paper(
            title=DEMO_PAPER_TITLE,
            topic=DEMO_PAPER_TOPIC,
            keywords=DEMO_PAPER_KEYWORDS,
            author_id=user.id,
            status=PaperStatus.DRAFT,
        )
        db.add(paper)
        await db.commit()
        await db.refresh(paper)

    # 延迟导入：避免 app.main 启动时形成 api → workflow → agents 的导入环
    from app.api.papers import _build_initial_state, _save_results
    from app.workflows.graph_builder import PaperWorkflowGraph

    # 用新建实例而不是全局 paper_workflow：全局实例在 import 时就绑定了当时的
    # settings，测试里改 DEMO_MODE 之后再用它，会拿着真实模型去打真接口。
    workflow = PaperWorkflowGraph()
    final_state = await workflow.run(_build_initial_state(paper))
    await _save_results(paper, final_state, db)
    await db.commit()
    await db.refresh(paper)
    logger.info("演示论文已生成：id=%s, 章节=%d, 评分=%s",
                paper.id, len(final_state.get("sections_content") or {}), paper.quality_score)
    return paper


async def ensure_demo_data(*, seed_paper: bool = True) -> dict:
    """幂等地准备好演示所需的一切。返回统计信息（便于启动日志与冒烟脚本核对）。"""
    if not settings.DEMO_MODE:
        return {"skipped": True, "reason": "DEMO_MODE 未开启"}

    vector_count = configure_demo_embeddings()

    async with AsyncSessionLocal() as db:
        user = await _ensure_user(db)
        paper = await _ensure_sample_paper(db, user) if seed_paper else None
        return {
            "vector_docs": vector_count,
            "user_email": user.email,
            "paper_id": paper.id if paper else None,
            "paper_status": paper.status.value if paper else None,
        }

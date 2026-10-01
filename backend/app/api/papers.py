# api/papers.py —— 论文管理 API：增删改查 + 触发生成工作流（已加认证）
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from sqlalchemy.orm import selectinload
from typing import Optional
import json

from app.database import get_db, AsyncSessionLocal
from app.models.paper import Paper, PaperStatus
from app.models.chapter import Chapter
from app.models.reference import Reference
from app.models.user import User
from app.schemas.paper import PaperCreate, PaperUpdate, PaperResponse, PaperDetail, PaperListResponse
from app.core.deps import get_current_user
from app.utils.llm import check_llm_config
from app.workflows.graph_builder import paper_workflow

router = APIRouter()

# ---------- 增删改查（CRUD，全部需要登录）----------

@router.post("/", response_model=PaperResponse, status_code=status.HTTP_201_CREATED)
async def create_paper(
    paper_data: PaperCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """创建新论文（自动挂到当前用户名下）"""
    paper = Paper(**paper_data.model_dump(), author_id=current_user.id)
    db.add(paper)
    await db.commit()
    await db.refresh(paper)
    return paper

@router.get("/", response_model=PaperListResponse)
async def list_papers(
    skip: int = 0,
    limit: int = 20,
    q: Optional[str] = None,
    status_filter: Optional[PaperStatus] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取当前用户自己的论文列表（支持搜索 / 状态筛选 / 分页）"""
    conditions = [Paper.author_id == current_user.id]
    if status_filter:
        conditions.append(Paper.status == status_filter)
    if q and q.strip():
        like = f"%{q.strip()}%"
        conditions.append(or_(
            Paper.title.ilike(like),
            Paper.topic.ilike(like),
            Paper.keywords.ilike(like),
        ))

    total = (await db.execute(
        select(func.count()).select_from(Paper).where(*conditions)
    )).scalar_one()

    query = select(Paper).where(*conditions).order_by(Paper.updated_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    return PaperListResponse(items=result.scalars().all(), total=total, skip=skip, limit=limit)

@router.get("/{paper_id}", response_model=PaperDetail)
async def get_paper(
    paper_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取论文详情（只能看自己的；别人的论文返回 404 而非 403，不泄露存在性）"""
    result = await db.execute(
        select(Paper)
        .options(selectinload(Paper.chapters), selectinload(Paper.references))
        .where(Paper.id == paper_id, Paper.author_id == current_user.id)
    )
    paper = result.scalar_one_or_none()
    if not paper:
        raise HTTPException(status_code=404, detail=f"论文 ID {paper_id} 不存在")
    return paper

@router.put("/{paper_id}", response_model=PaperResponse)
async def update_paper(
    paper_id: int,
    paper_data: PaperUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """更新论文（只更新传了的字段，且只能是自己的）"""
    result = await db.execute(
        select(Paper).where(Paper.id == paper_id, Paper.author_id == current_user.id)
    )
    paper = result.scalar_one_or_none()
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")

    for key, value in paper_data.model_dump(exclude_unset=True).items():
        setattr(paper, key, value)

    await db.commit()
    await db.refresh(paper)
    return paper

@router.delete("/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_paper(
    paper_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """删除论文（章节和文献级联删除）"""
    result = await db.execute(
        select(Paper).where(Paper.id == paper_id, Paper.author_id == current_user.id)
    )
    paper = result.scalar_one_or_none()
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")
    await db.delete(paper)
    await db.commit()

# ---------- 工作流结果持久化（和之前完全一样）----------

async def _save_results(paper: Paper, final_state: dict, db: AsyncSession):
    """把工作流产出存进数据库：全文、大纲、章节、文献"""
    paper.content = final_state.get("final_paper", "")
    paper.outline = json.dumps(final_state.get("outline") or {}, ensure_ascii=False)
    # 质量评审结果持久化，刷新/重进页面后评审卡片不消失
    paper.quality_score = final_state.get("quality_score")
    paper.quality_comment = final_state.get("quality_comment")
    paper.quality_dimensions = json.dumps(final_state.get("quality_dimensions") or [], ensure_ascii=False)
    paper.status = PaperStatus.COMPLETED if final_state.get("final_paper") else PaperStatus.DRAFT

    old_chapters = (await db.execute(
        select(Chapter).where(Chapter.paper_id == paper.id)
    )).scalars().all()
    for old in old_chapters:
        await db.delete(old)

    for order, (title, content) in enumerate((final_state.get("sections_content") or {}).items()):
        db.add(Chapter(paper_id=paper.id, title=title, content=content, order_index=order))

    old_refs = (await db.execute(
        select(Reference).where(Reference.paper_id == paper.id)
    )).scalars().all()
    existing_titles = {r.title for r in old_refs}

    for doc in final_state.get("references") or []:
        title = doc.get("title")
        if not title or title in existing_titles:
            continue
        year = doc.get("year")
        db.add(Reference(
            paper_id=paper.id,
            title=title,
            authors=doc.get("authors"),
            journal=doc.get("journal"),
            year=int(year) if year else None,
        ))
        existing_titles.add(title)

def _build_initial_state(paper: Paper) -> dict:
    """把数据库里的论文记录组装成工作流初始状态"""
    return {
        "paper_id": paper.id,
        "title": paper.title,
        "topic": paper.topic or "",
        "keywords": [k.strip() for k in paper.keywords.split(",")] if paper.keywords else [],
        "is_topic_clear": True,
        "revision_count": 0,
        "max_revisions": 2,
        "sections_content": {},
    }

# ---------- 触发生成（同样需要登录 + 归属校验）----------

@router.post("/{paper_id}/generate")
async def generate_paper(
    paper_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """同步生成：等整个工作流跑完才返回"""
    result = await db.execute(
        select(Paper).where(Paper.id == paper_id, Paper.author_id == current_user.id)
    )
    paper = result.scalar_one_or_none()
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")

    check_llm_config()

    final_state = await paper_workflow.run(_build_initial_state(paper))

    await _save_results(paper, final_state, db)
    await db.commit()

    return {
        "message": "论文生成完成",
        "paper_id": paper_id,
        "quality_score": final_state.get("quality_score"),
        "sections": len(final_state.get("sections_content", {})),
        "references": len(final_state.get("references", [])),
    }

@router.post("/{paper_id}/stream-generate")
async def stream_generate_paper(
    paper_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """流式生成（SSE）：每个节点完成就推一条进度"""
    result = await db.execute(
        select(Paper).where(Paper.id == paper_id, Paper.author_id == current_user.id)
    )
    paper = result.scalar_one_or_none()
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")

    check_llm_config()

    initial_state = _build_initial_state(paper)

    async def event_generator():
        final_state = dict(initial_state)
        try:
            async for event in paper_workflow.stream(initial_state):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                for update in event.values():
                    final_state.update(update)

            async with AsyncSessionLocal() as session:
                paper_obj = (await session.execute(
                    select(Paper).where(Paper.id == paper_id)
                )).scalar_one()
                await _save_results(paper_obj, final_state, session)
                await session.commit()

            yield f"data: {json.dumps({'done': True, 'quality_score': final_state.get('quality_score'), 'quality_comment': final_state.get('quality_comment'), 'quality_dimensions': final_state.get('quality_dimensions', [])}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
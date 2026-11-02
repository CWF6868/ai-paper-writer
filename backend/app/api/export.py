# api/export.py —— 论文导出接口：把论文导出为 Word / PDF 文件（需登录且只能导出自己的）
import asyncio
from io import BytesIO
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.paper import Paper
from app.models.user import User
from app.core.deps import get_current_user
from app.utils.exporter import build_docx, build_pdf

router = APIRouter()


@router.get("/{paper_id}/export")
async def export_paper(
    paper_id: int,
    format: str = Query(..., description="docx 或 pdf"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    fmt = format.strip().lower()
    if fmt not in ("docx", "pdf"):
        raise HTTPException(status_code=400, detail="format 只支持 docx 或 pdf")

    result = await db.execute(
        select(Paper)
        .options(selectinload(Paper.references))
        .where(Paper.id == paper_id, Paper.author_id == current_user.id)
    )
    paper = result.scalar_one_or_none()
    if paper is None:
        raise HTTPException(status_code=404, detail="论文不存在")

    try:
        # P0-2：python-docx / fpdf2 是纯 CPU 的同步库，几万字的论文要几百毫秒到数秒。
        # 直接在协程里调，整个事件循环会被占住 —— 同一进程里其他人的 SSE 生成进度、
        # 列表接口全部卡顿。丢进线程池后，导出让出事件循环，单次导出的绝对耗时不变
        # （CPU 密集本就无法靠线程加速），换来的是「导出期间服务仍可响应」。
        if fmt == "docx":
            # 先把 relationship 取出来：确认它已随 selectinload 加载完成，
            # 避免线程里触发惰性加载（跨线程访问 session 会直接报错）
            list(paper.references)
            data = await asyncio.to_thread(build_docx, paper)
            media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            suffix = "docx"
        else:
            list(paper.references)
            data = await asyncio.to_thread(build_pdf, paper)
            media = "application/pdf"
            suffix = "pdf"
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    filename = quote((paper.title or "论文").strip()[:40])
    headers = {
        "Content-Disposition": f"attachment; filename*=UTF-8''{filename}.{suffix}"
    }
    return StreamingResponse(BytesIO(data), media_type=media, headers=headers)
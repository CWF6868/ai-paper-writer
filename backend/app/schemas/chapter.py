# chapter.py —— 章节相关的 Pydantic 模式
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List
from datetime import datetime


class ChapterBase(BaseModel):
    """章节公共字段"""
    title: str = Field(..., min_length=1, max_length=200, description="章节标题")
    content: Optional[str] = None
    summary: Optional[str] = None


class ChapterCreate(ChapterBase):
    """创建章节时的额外字段"""
    paper_id: int
    parent_id: Optional[int] = None
    order_index: int = 0


class ChapterOut(ChapterBase):
    """返回给前端的章节信息"""
    id: int
    paper_id: int
    parent_id: Optional[int]
    order_index: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChapterDetail(ChapterOut):
    """带子章节的完整章节（用于树形展示）"""
    children: List["ChapterOut"] = []


ChapterDetail.model_rebuild()  # 让 Pydantic 解析 children 中对 ChapterOut 的自引用
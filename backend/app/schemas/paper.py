
# schemas/paper.py —— 论文相关接口的 Pydantic 模式（v2）
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime

from app.models.paper import PaperStatus

class PaperBase(BaseModel):
    """论文基础模式"""
    title: str = Field(..., min_length=5, max_length=500, description="论文标题")
    topic: Optional[str] = Field(None, max_length=200, description="选题方向")
    keywords: Optional[str] = Field(None, max_length=500, description="关键词，逗号分隔")
    abstract: Optional[str] = Field(None, description="摘要")

class PaperCreate(PaperBase):
    """创建论文用"""
    pass

class PaperUpdate(BaseModel):
    """更新论文用（全部可选，只更新传了的字段）"""
    title: Optional[str] = Field(None, min_length=5, max_length=500)
    topic: Optional[str] = None
    keywords: Optional[str] = None
    abstract: Optional[str] = None
    content: Optional[str] = None
    outline: Optional[str] = None
    status: Optional[PaperStatus] = None

class PaperResponse(PaperBase):
    """论文列表的响应模式"""
    id: int
    status: PaperStatus
    outline: Optional[str] = None
    # 认证功能上线前，论文可以没有作者，所以必须是可选
    author_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ChapterBrief(BaseModel):
    """章节简要信息"""
    id: int
    title: str
    order_index: int

    model_config = ConfigDict(from_attributes=True)

class ReferenceBrief(BaseModel):
    """文献简要信息"""
    id: int
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)

class PaperDetail(PaperResponse):
    """论文详情模式（比列表多：全文、章节、文献、质量评审）"""
    content: Optional[str] = None
    chapters: List[ChapterBrief] = []
    references: List[ReferenceBrief] = []
    quality_score: Optional[float] = None
    quality_comment: Optional[str] = None
    quality_dimensions: Optional[str] = None   # JSON 文本，前端自行解析


class PaperListResponse(BaseModel):
    """论文列表分页响应：items + 总数"""
    items: List[PaperResponse]
    total: int
    skip: int
    limit: int
# schemas/topic.py —— 选题推荐接口的 Pydantic 模式
from pydantic import BaseModel, Field
from typing import Optional


class TopicRecommendRequest(BaseModel):
    """选题推荐入参"""
    field: str = Field(..., min_length=1, max_length=200, description="研究方向")
    keywords: Optional[str] = Field(None, max_length=500, description="关键词，逗号分隔")
    requirements: Optional[str] = Field(None, max_length=500, description="特殊要求")


class TopicRecommendResponse(BaseModel):
    """选题推荐出参：topics 为 markdown 文本"""
    topics: str
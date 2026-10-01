# api/topics.py —— 选题推荐：复用 TopicAgent，暴露给前端
from fastapi import APIRouter, Depends, HTTPException

from app.agents.topic_agent import TopicAgent
from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.topic import TopicRecommendRequest, TopicRecommendResponse
from app.utils.llm import check_llm_config

router = APIRouter()

# 单例复用，避免每次请求都重建 LLM 客户端
topic_agent = TopicAgent()


@router.post("/recommend", response_model=TopicRecommendResponse)
async def recommend_topic(
    data: TopicRecommendRequest,
    current_user: User = Depends(get_current_user),
):
    """根据研究方向/关键词推荐 3-5 个选题（需登录，只依赖 DeepSeek）"""
    check_llm_config(require_embedding=False)

    field = data.field.strip()
    if not field:
        raise HTTPException(status_code=400, detail="研究方向不能为空")

    result = await topic_agent.execute({
        "field": field,
        "keywords": data.keywords or "",
        "requirements": data.requirements or "",
    })
    return TopicRecommendResponse(topics=result.get("topics", ""))
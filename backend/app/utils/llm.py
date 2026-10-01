# utils/llm.py —— 统一的模型接入层（大脑 + 向量化）
from typing import Optional

from fastapi import HTTPException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings

def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    streaming: bool = False,
):
    """
    返回一个大模型(LLM)实例，供所有 Agent 使用。
    provider 决定用哪家：deepseek（本系统已配置）。
    """
    provider = provider or settings.LLM_PROVIDER

    if provider == "deepseek":
        return ChatOpenAI(
            model=model or settings.DEEPSEEK_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            temperature=temperature,
            streaming=streaming,
        )

    raise ValueError(f"不支持的模型提供商: {provider}（当前仅配置了 deepseek）")

def get_embedding_model():
    """
    返回一个向量化(Embedding)模型实例，供 RAG 使用。
    关键参数 check_embedding_ctx_length=False：
    langchain 默认会先把文本转成 token 编号数组再发送（OpenAI 官方接口支持），
    但百炼兼容接口只接受纯文本字符串，必须关掉这个预处理，否则报 400。
    """
    return OpenAIEmbeddings(
        model=settings.DASHSCOPE_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        check_embedding_ctx_length=False,
    )


def check_llm_config(require_embedding: bool = True):
    """生成前校验 Key 是否已配置。require_embedding=False 时只校验 DeepSeek（如选题推荐）。"""
    missing = []
    if not settings.DEEPSEEK_API_KEY:
        missing.append("DeepSeek 的 DEEPSEEK_API_KEY")
    if require_embedding and not settings.DASHSCOPE_API_KEY:
        missing.append("DashScope 的 DASHSCOPE_API_KEY")
    if missing:
        raise HTTPException(
            status_code=400,
            detail="未配置 " + "、".join(missing) + "，请在 backend/.env 中填写后重启后端",
        )
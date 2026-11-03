# utils/llm.py —— 模型接入层的对外门面
#
# P0-3 起，ChatOpenAI 的构造已收口到 app/core/llm_client.py（统一管理超时、
# 重试、用量计量回调）。这里只保留原有函数签名，避免全项目 import 改动，
# 同时保证「只有一个地方能造模型」——否则超时/回调必然分叉，成本就统计不全。
from typing import Optional

from fastapi import HTTPException

from app.config import settings
from app.core.llm_client import build_chat_model, build_embedding_model
from app.core.llm_client import CostTrackingCallback  # noqa: F401  供需要显式挂载的调用方使用


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    streaming: bool = False,
    max_tokens: Optional[int] = None,
):
    """
    返回一个大模型(LLM)实例，供所有 Agent 使用。
    provider 决定用哪家：deepseek（本系统已配置）。

    max_tokens 默认取 settings.LLM_MAX_TOKENS：不显式设置时会落到供应商默认值
    （DeepSeek 为 4096），长文评审的结构化输出会被截断成非法 JSON，进而触发
    三级降级、整篇评审作废。这是真实故障，不是保守起见。
    """
    provider = provider or settings.LLM_PROVIDER

    if provider == "deepseek":
        return build_chat_model(
            model=model,
            temperature=temperature,
            streaming=streaming,
            max_tokens=max_tokens,
        )

    raise ValueError(f"不支持的模型提供商: {provider}（当前仅配置了 deepseek）")


def get_embedding_model():
    """
    返回一个向量化(Embedding)模型实例，供 RAG 使用。
    关键参数 check_embedding_ctx_length=False 由 llm_client 统一设置：
    langchain 默认会先把文本转成 token 编号数组再发送（OpenAI 官方接口支持），
    但百炼兼容接口只接受纯文本字符串，必须关掉这个预处理，否则报 400。
    """
    return build_embedding_model()


def check_llm_config(require_embedding: bool = True):
    """生成前校验 Key 是否已配置。require_embedding=False 时只校验 DeepSeek（如选题推荐）。

    演示模式直接放行：演示实例本来就没有（也不该有）真实 Key，
    模型与向量化都在本地完成，这里再拦一次只会把 Demo 拦成 400。
    """
    if settings.DEMO_MODE:
        return

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

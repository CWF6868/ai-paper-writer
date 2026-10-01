"""向量化入口"""
# embeddings.py —— 向量化入口
from app.utils.llm import get_embedding_model

def get_embeddings():
    """
    获取向量化模型。
    统一复用 utils/llm.py 里的 get_embedding_model()，
    保证全系统只有一个向量化入口：百炼 text-embedding-v3（OpenAI 兼容接口）。
    以后想换向量化模型，只改 llm.py 一处即可。
    """
    return get_embedding_model()
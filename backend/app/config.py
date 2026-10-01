from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
"""读取.env的配置"""
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    APP_NAME: str = "AI 论文写作系统"
    DEBUG: bool = False

    # ---- 数据库 ----
    DATABASE_URL: str = "sqlite:///./paper_writer.db"

    # ---- LLM 大脑（DeepSeek）----
    LLM_PROVIDER: str = "deepseek"
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # ---- Embedding 向量化（阿里云百炼 text-embedding-v3）----
    EMBEDDING_PROVIDER: str = "dashscope"
    DASHSCOPE_API_KEY: Optional[str] = None
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DASHSCOPE_MODEL: str = "text-embedding-v3"

    # ---- 向量库 ----
    VECTOR_DB_PATH: str = "./data/chroma"

    # ---- JWT ----
    SECRET_KEY: str = "dev-secret-change-me"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

settings = Settings()
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

    # ---- 模型输出上限 ----
    # 评审要输出 5 个维度的点评 + 原文证据 + 分章建议，长文评审的输出常超过
    # 供应商默认上限（DeepSeek 默认 4096），导致 function calling 的 JSON 参数
    # 被截断成非法 JSON → 三级降级全失败 → 整篇评审作废（2026-11-02 实测到）。
    # DeepSeek 支持最大 8192，这里取满；可用 .env 的 LLM_MAX_TOKENS 覆盖。
    LLM_MAX_TOKENS: int = 8192

    # ---- 向量库 ----
    VECTOR_DB_PATH: str = "./data/chroma"

    # ---- 并发控制（P0-2）----
    # 章节写作并发上限：同一篇论文、同一写作轮次内最多同时发出几个写作请求。
    # 为什么是 4：写作是纯网络等待（单章 6-10s），串行写 8 章就是 8 倍等待；
    # 4 既能把「8 章」压成 2 批，又低于 DeepSeek 单账号的常见并发容忍度。
    # 注意这是「每篇」的上限：K 篇论文同时生成时，进程内总在途请求 ≈ K × 该值，
    # 多用户场景应按账号配额调小（.env 的 SECTION_CONCURRENCY）。
    SECTION_CONCURRENCY: int = 4

    # LLM 请求失败重试次数：并发抬高后偶发 429/5xx 的概率上升，
    # 不重试的话一次限流就会让整章（并发下甚至整篇）报废。
    # langchain-openai 会按指数退避重试，这里只控制次数。
    LLM_MAX_RETRIES: int = 3

    # ---- LLM 调用治理（P0-3 成本故事）----
    # 单次请求超时。为什么要设：不设时走 SDK 默认（600s），一次卡住的调用会让
    # 整篇生成挂到用户以为页面死了；并发下还会一直占着 Semaphore 额度不释放。
    # 60s 的依据：实测单章写作 6-10s、长文评审最长约 40s，60s 已留足 1.5 倍余量。
    LLM_TIMEOUT_SECONDS: float = 60.0

    # 峰谷计价开关。DeepSeek 自 2026-08-17 起按峰谷双价计费，
    # 峰时（北京时间工作日 09:00-12:00、14:00-18:00）价格是空闲时段的 2 倍。
    # 关掉则一律按空闲价估算（只影响 cost 字段的估算值，不影响 tokens 统计）。
    LLM_PEAK_PRICING: bool = True

    # ---- 质量控制 ----
    # 质量检查通过线：评审分 >= 该值才结束，否则打回重写。
    # 0.44 的依据（2026-11-02 端到端实测，6 篇真实生成 + 首检分观测）：
    #   首检分实测 0.42 / 0.42 / 0.44 / 0.44 / 0.44 / 0.49（中位数 0.44，
    #   重写后 0.42-0.51）。按数据推演：
    #     <=0.42 → 重写率 0%，门禁失效；
    #     0.43-0.44 → 重写率 33%，平均质量 0.445；
    #     0.45-0.49 → 重写率 83%，平均质量 0.457；
    #     >=0.50 → 重写率 100%，门禁退化成固定流程。
    #   取 0.44 = 实测中位数：中位数以上直接放行，以下才重写。
    #   质量差异（-0.015）在评分噪声带（0.02）内，换来重写率从 100% 降到 33%。
    # 标定方法：scripts/analyze_threshold.py（用实测首检分/重写后分推演每个阈值的后果）。
    # 可用 .env 的 QUALITY_PASS_THRESHOLD 覆盖。
    QUALITY_PASS_THRESHOLD: float = 0.44

    # ---- 演示模式（公网 Demo）----
    # 打开后：模型调用换成本地确定性实现，向量化换成哈希向量，不发起任何外部请求。
    # 为什么需要它：公网 Demo 若挂真实 Key，会被刷成本；不挂 Key 则生成接口直接 400。
    # 注意：这里只替换「模型输出」这一层，RAG/并发/质量门禁/SSE/导出仍走生产代码。
    DEMO_MODE: bool = False
    # 启动时自动准备演示数据（语料 + 演示账号 + 一篇成稿）。首次冷启动会多花几秒。
    DEMO_SEED_ON_START: bool = True
    # 演示节奏（毫秒）：SSE 进度事件之间的最小间隔。
    # 默认 0 = 不干预。演示模式本身没有推理耗时，四个阶段会在 0.1 秒内全部推完，
    # 进度条一闪而过、看起来像假动作；录 GIF 时设 600-900 能让过程可读。
    # 只在 DEMO_MODE 下生效，真实模式不受影响。
    DEMO_PACE_MS: int = 0

    # ---- JWT ----
    SECRET_KEY: str = "dev-secret-change-me"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

settings = Settings()
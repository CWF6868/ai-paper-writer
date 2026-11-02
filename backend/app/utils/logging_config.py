# logging_config.py —— 全局日志配置（P0-1 观测链路的必要条件）
#
# 背景：项目原先没有任何 logging 配置。
# Python 的 lastResort handler 只输出 WARNING+，且 **不渲染 extra 字段**，
# 结果是 app/llm/structured.py 里所有 LLM_CALL / TOOL_CALL 的 INFO 日志
# 被直接丢弃，logger.error("reviewer degraded", extra={"error": ...}) 的
# 失败原因也一起丢失 —— 「每级降级留痕」成了空承诺（2026-11-01 评测实测：
# 6 篇全跑，LLM_CALL/TOOL_CALL 日志 0 条）。
#
# 本模块提供 setup_logging()，由 app/__init__.py 在包导入时调用一次，
# 从而覆盖 FastAPI、脚本、pytest 三条执行路径。
import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str | None = None, stream=None) -> None:
    """配置根 logger。幂等：重复调用只调整级别，不重复添加 handler。

    level 取值优先级：显式参数 > 环境变量 LOG_LEVEL > INFO。
    pytest 场景建议设 LOG_LEVEL=WARNING 以避免日志刷屏。
    """
    resolved = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    numeric = getattr(logging, resolved, logging.INFO)

    root = logging.getLogger()
    if getattr(root, "_ai_paper_configured", False):
        root.setLevel(numeric)
        return

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(numeric)
    # 标记已配置，避免 uvicorn reload / 多次 import 时叠加 handler
    root._ai_paper_configured = True  # type: ignore[attr-defined]

    # 第三方库降噪：HTTP 请求日志量大且无诊断价值
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

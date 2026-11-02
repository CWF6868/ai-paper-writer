# app/__init__.py —— 包入口：导入即完成日志配置
#
# 放在包入口而不是 main.py，是为了同时覆盖三条执行路径：
#   FastAPI（uvicorn app.main:app）、脚本（scripts/*.py）、pytest。
# 只配置 logging，不引入业务依赖，因此没有循环导入风险。
from app.utils.logging_config import setup_logging

setup_logging()

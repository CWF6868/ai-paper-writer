# llm/__init__.py —— 结构化输出 + 工具调用层（P0-1 改造）
#
# 为什么独立成包而不塞进 agents/：
# Schema 是跨层契约——API 层要序列化、前端要渲染、评测脚本要断言、测试要构造 fixture。
# 放进某个 Agent 文件里，必然被反向 import。
#
# 目录职责：
#   schemas.py     所有结构化输出模型（单一事实来源）
#   structured.py  with_structured_output 封装：三级降级 + 重试 + 观测日志
#   tools/         工具调用层（registry 注册 + 预算控制）

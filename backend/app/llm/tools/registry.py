# registry.py —— 工具注册与预算控制
#
# 预算写死在这里并被单测锁定（test_tool_budget）：
# 把「成本控制」写成测试，而不是嘴上说说。
from langchain_core.tools import BaseTool

from app.llm.tools.citation import verify_citation
from app.llm.tools.literature import search_literature

# 硬上限：防止模型无限查证（每次工具调用都有真实延迟与 token 成本）
MAX_TOOL_ROUNDS = 2      # 最多 2 轮「模型→工具→模型」往返
MAX_TOOL_CALLS = 6       # 单次评审最多 6 次工具调用（跨轮累计）

# 评审 Agent 的工具箱：2 个，不多不少。
# 只挂「LLM 自己产不出信息」的工具；字数/重复度等三行 Python 能算的
# 在节点里算好塞 context，不挂成工具。
REVIEWER_TOOLS: list[BaseTool] = [verify_citation, search_literature]

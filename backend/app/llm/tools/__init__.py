# tools/__init__.py —— 工具调用层
#
# 工具三判据（全部满足才做成工具）：
#   ① LLM 自己产不出这个信息（需要访问外部事实）
#   ② 不能用三行 Python 算出来（能算的在代码里算好塞进 context）
#   ③ 无副作用（有写操作的不进 Agent 自动循环）
#
# 本包只挂 2 个工具：verify_citation（查证引用）、search_literature（文献内检索）。
# text.py 是纯计算工具函数——不给模型调，是给 verify_citation 内部用的。

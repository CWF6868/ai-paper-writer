# context.py —— 工具上下文注入
#
# 工具需要访问「本次评审的文献列表」，但工具函数签名里不该带这个参数
# （模型不会填）。用 contextvars 在跑图前注入。
#
# 为什么不用全局变量：并发章节写作（P0-2 改造后）多个协程共享
# 全局变量会串数据；contextvars 是协程隔离的。
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List

# 本次评审/写作流程可用的参考文献列表（语料含正文 text 字段）
_current_references: ContextVar[List[Dict[str, Any]]] = ContextVar(
    "current_references", default=[]
)


@contextmanager
def set_current_references(refs: List[Dict[str, Any]]) -> Iterator[None]:
    """在 with 块内注入文献列表，退出时恢复（graph_builder._quality_check 使用）。"""
    token = _current_references.set(list(refs or []))
    try:
        yield
    finally:
        _current_references.reset(token)

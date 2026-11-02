# literature.py —— 文献内语义检索工具
#
# 语义决策（设计文档两处描述矛盾，此处取「检索当前论文参考文献」）：
#   1. 返回的 index 必须能直接用于 verify_citation（文档硬约束），
#      只有在「本文参考文献列表」内检索，编号才与 verify_citation 同源。
#   2. 评审 Prompt 里只有文献题录（[n] 作者. 题名[J]...），看不到语料正文；
#      在语料正文里做语义检索是「LLM 自己产不出的信息」，符合工具三判据。
#   3. 查全库找新文献对本流水线不可操作：评审无权增删文献，writer 只消费已检索的 5 条。
import asyncio
import hashlib
import time
from typing import List

from langchain_core.tools import tool

from app.llm.structured import log_tool_call
from app.llm.tools import citation
from app.llm.tools.context import _current_references
from app.llm.tools.text import cosine


async def _search(query: str, top_k: int) -> List[dict]:
    t0 = time.perf_counter()
    ok = False
    try:
        refs = _current_references.get()
        if not refs or not (query or "").strip():
            return []
        # embed_query 是同步 HTTP 调用，放线程池避免阻塞事件循环
        q_vec = await asyncio.to_thread(citation._embed, query)
        scored = []
        for i, ref in enumerate(refs, start=1):
            text = ref.get("text", "") or ""
            if not text:
                continue
            sim = cosine(q_vec, await asyncio.to_thread(citation._embed, text))
            if sim > 0.30:  # 低于 0.3 视为无关，不占用返回名额
                scored.append({
                    "index": i,
                    "title": ref.get("title") or f"文献[{i}]",
                    "score": round(sim, 3),
                })
        scored.sort(key=lambda x: -x["score"])
        ok = True
        return scored[:max(1, top_k)]
    finally:
        digest = hashlib.md5((query or "")[:50].encode()).hexdigest()[:10]
        log_tool_call("reviewer", "search_literature", ok, (time.perf_counter() - t0) * 1000,
                      args_digest=digest)


@tool
async def search_literature(query: str, top_k: int = 3) -> List[dict]:
    """在「本文参考文献」的语料正文里做语义检索。

    返回 [{index, title, score}]，按相关度降序；
    index 即参考文献编号，可直接传给 verify_citation 查证。

    args:
        query: 检索主题，如"某章缺少支撑时想确认哪条文献讨论过 X"
        top_k: 最多返回几条，默认 3
    """
    return await _search(query, top_k)

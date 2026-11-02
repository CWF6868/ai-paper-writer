# citation.py —— 引文查证工具（评审 Agent 的「证据之眼」）
#
# 工具内部默认不调 LLM：先做确定性初筛（embedding 余弦 + 关键词重合），
# 只有落在 [0.35, 0.72] 灰区才升级到 LLM 裁决。按实测灰区占比 15-25%，
# 意味着约 80% 的查证是纯向量计算，<100ms，不增加 LLM 往返。
#
# 若每个工具内部都套一次 LLM，评审一轮 3 次工具调用 = 3 次额外 LLM 往返，
# 单篇耗时从 119s 涨到 150s+，工具调用就变成负优化了。
import asyncio
import hashlib
import time
from functools import lru_cache
from typing import List

from langchain_core.tools import tool

from app.llm.schemas import CitationVerdict
from app.llm.structured import StructuredOutputError, call_structured, log_tool_call
from app.llm.tools.context import _current_references
from app.llm.tools.text import cosine, keyword_jaccard

# 初筛阈值：>=0.72 且关键词重合 >=0.30 判 supported；<=0.35 判 unsupported；中间灰区请 LLM
SUPPORTED_SIM = 0.72
UNSUPPORTED_SIM = 0.35
SUPPORTED_OVERLAP = 0.30


@lru_cache(maxsize=512)
def _embed_cached(text: str) -> List[float]:
    """文本向量化，带进程内缓存（同一文献正文反复查证不重复请求 embedding 接口）。"""
    from app.rag.embeddings import get_embeddings
    return list(get_embeddings().embed_query(text))


def _embed(text: str) -> List[float]:
    """单条向量化（测试通过 monkeypatch 本函数注入假向量）。"""
    return _embed_cached(text)


async def _llm_judge(claim: str, ref_text: str, sim: float) -> CitationVerdict:
    """灰区裁决：请 LLM 判断「文献是否支持该论断」。

    裁决本身是一次结构化输出（CitationVerdict），复用三级降级；
    裁决失败时保守判 partial，不阻塞评审主流程。
    """
    from app.utils.llm import get_llm

    llm = get_llm(temperature=0.0)
    messages = [
        ("system",
         "你是学术引文查证助手。给定一条论文中的论断和一篇参考文献的正文片段，"
         "判断该文献是否支持这条论断。无法确定时判 partial，不要过度猜。"),
        ("human",
         f"参考文献正文片段：\n{ref_text[:1500]}\n\n"
         f"待查证论断：{claim}\n\n"
         f"（向量初筛相似度 {sim:.2f}，仅供参考）\n"
         "请给出裁决。"),
    ]
    return await call_structured(llm, CitationVerdict, messages, node="citation_judge")


async def _verify(ref_index: int, claim: str) -> dict:
    t0 = time.perf_counter()
    ok = False
    try:
        refs = _current_references.get()
        if not (1 <= ref_index <= len(refs)):
            verdict = CitationVerdict(verdict="not_found", similarity=0.0,
                                     reason=f"文献编号越界（共 {len(refs)} 条）")
        else:
            ref_text = (refs[ref_index - 1].get("text", "") or "")
            if not ref_text:
                verdict = CitationVerdict(verdict="not_found", similarity=0.0,
                                         reason="该文献无语料正文")
            else:
                # ── 第一层：确定性初筛（不调 LLM）──
                # embed_query 是同步 HTTP 调用，直接调会阻塞事件循环
                # （并发生成多篇论文时互相拖累），放线程池执行
                claim_vec, ref_vec = await asyncio.gather(
                    asyncio.to_thread(_embed, claim),
                    asyncio.to_thread(_embed, ref_text),
                )
                sim = cosine(claim_vec, ref_vec)
                overlap = keyword_jaccard(claim, ref_text)
                if sim >= SUPPORTED_SIM and overlap >= SUPPORTED_OVERLAP:
                    verdict = CitationVerdict(
                        verdict="supported", similarity=round(sim, 3),
                        reason=f"语义相似度 {sim:.2f}，关键词重合 {overlap:.2f}，文献支持该论断",
                    )
                elif sim <= UNSUPPORTED_SIM:
                    verdict = CitationVerdict(
                        verdict="unsupported", similarity=round(sim, 3),
                        reason=f"语义相似度仅 {sim:.2f}，文献与论断无关",
                    )
                else:
                    # ── 第二层：灰区才请 LLM 裁决（控成本的关键）──
                    try:
                        verdict = await _llm_judge(claim, ref_text, sim)
                    except StructuredOutputError as e:
                        verdict = CitationVerdict(
                            verdict="partial", similarity=round(sim, 3),
                            reason=f"LLM 裁决不可用（{e}），按相似度保守判 partial",
                        )
        ok = True
        return verdict.model_dump()
    finally:
        digest = hashlib.md5(f"{ref_index}:{claim[:50]}".encode()).hexdigest()[:10]
        log_tool_call("reviewer", "verify_citation", ok, (time.perf_counter() - t0) * 1000,
                      args_digest=digest)


@tool
async def verify_citation(ref_index: int, claim: str) -> dict:
    """校验正文中的论断是否被第 ref_index 条参考文献支持。

    args:
        ref_index: 参考文献编号（正文里的 [n] 的 n，从 1 开始）
        claim: 需要核验的论断原文（一句话）
    """
    return await _verify(ref_index, claim)

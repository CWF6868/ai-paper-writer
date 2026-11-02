# text.py —— 纯文本计算（不是给模型调的工具，是给 verify_citation 内部用的）
#
# 设计哲学：能用三行 Python 算出来的（字数、重复度、关键词重合），
# 绝不做成 LLM 工具——多一次 LLM 往返就多一份出错概率和延迟。
import re
from typing import Dict, List

_WORD_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fa5]")


def _keywords(text: str) -> set:
    """中英混合分词：英文按单词、中文按单字（简单但对短文本重合度判断够用）。"""
    return set(_WORD_RE.findall((text or "").lower()))


def _ngrams(text: str, n: int = 2) -> set:
    """中文 bigram + 英文单词混合 n-gram 集合。"""
    tokens = _WORD_RE.findall((text or "").lower())
    grams: set = set()
    for t in tokens:
        if len(t) > 1:
            for i in range(len(t) - n + 1):
                grams.add(t[i:i + n])
        else:
            grams.add(t)
    return grams


def keyword_jaccard(a: str, b: str) -> float:
    """关键词重合度（Jaccard）：两段文本的关键词交集/并集。"""
    ka, kb = _keywords(a), _keywords(b)
    if not ka or not kb:
        return 0.0
    return len(ka & kb) / len(ka | kb)


def ngram_jaccard(a: str, b: str, n: int = 2) -> float:
    """n-gram 重合度：比单字更抗「同一批字洗牌后重复」的伪重复。"""
    ga, gb = _ngrams(a, n), _ngrams(b, n)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def cosine(u: List[float], v: List[float]) -> float:
    """向量余弦相似度。长度不齐时按短的截断（防御）。"""
    n = min(len(u), len(v))
    if n == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(u[:n], v[:n]))
    nu = sum(a * a for a in u[:n]) ** 0.5
    nv = sum(b * b for b in v[:n]) ** 0.5
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def section_overlap_report(sections: Dict[str, str], threshold: float = 0.45) -> List[Dict]:
    """章节两两重复度（n-gram Jaccard），识别「引言和方法重复」这类通病。

    返回超过阈值的章节对：[{a, b, overlap}]，按重复度降序。
    纯计算，给评审节点算好塞进 context 用，不做成 LLM 工具。
    """
    titles = list(sections.keys())
    report = []
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            ov = ngram_jaccard(sections[titles[i]], sections[titles[j]])
            if ov >= threshold:
                report.append({"a": titles[i], "b": titles[j], "overlap": round(ov, 3)})
    return sorted(report, key=lambda x: -x["overlap"])

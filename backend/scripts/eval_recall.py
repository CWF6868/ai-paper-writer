#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RAG 检索命中评测（分级 Recall + MRR + 构造性归因）—— 纯离线、只调向量化、不调生成模型。

【为什么不止报 Recall@5】
本评测集是"用文献原词造查询"的自建集，Recall@5 会稳定 100%，属于构造性结果，
单独报 100% 不具备说服力、且经不起追问。因此本脚本同时给出：
    - Recall@1 / Recall@3 / Recall@K    分级阈值，K 越大越宽松
    - MRR（平均倒数排名）                 对排序质量更敏感
    - 命中排名分布                        看命中是排第 1 还是靠后（暴露任务难度）
    - 查询→文献 词汇重合度归因            量化"查询是否抄自文献原文"
只有把"为什么这么高"讲清楚，指标才可信。对外报数请优先报 Recall@1 与 MRR。

【口径】
    - 语料：app/rag/corpus.py 固定 42 篇
    - 每个查询预先标注「期望命中的文献 id」，用生产向量库按语义相似度取 top-K
    - Recall@n = 命中且排名 ≤ n 的期望文献数 / 期望文献总数
    - MRR      = mean(1 / 该期望文献的排名)，未命中记 0

用法（在 backend/ 目录下执行）：
    python scripts/eval_recall.py                        # 默认 K=5，完整重跑检索
    python scripts/eval_recall.py --k 5 --out data/eval_recall.json
    python scripts/eval_recall.py --derive-only          # 不调 embedding，从已有 JSON 重算派生指标
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.vector_store import vector_store  # noqa: E402

# ---------------------------------------------------------------------------
# 评测集：23 条查询 / 27 条期望文献
# 标注原则：每条查询的期望文献在语料里是「无歧义的主题对应项」，
#          例如「检索增强生成 缓解幻觉」唯一对应 ref-010（RAG 原论文）。
# ⚠️ 注意：查询词取自文献原文，因此本集测的是「检索链路是否通畅」，
#          不是「检索器对新表述的泛化能力」。解读时务必带上这句话。
# ---------------------------------------------------------------------------
EVAL_CASES: List[Dict] = [
    {"query": "Transformer 自注意力机制 序列建模", "expected": ["ref-001"]},
    {"query": "BERT 双向编码器 掩码语言模型 预训练", "expected": ["ref-002"]},
    {"query": "大语言模型 少样本学习 上下文学习", "expected": ["ref-004"]},
    {"query": "人类反馈强化学习 指令微调 对齐", "expected": ["ref-005"]},
    {"query": "缩放定律 模型参数量与训练数据量最优配比", "expected": ["ref-007"]},
    {"query": "检索增强生成 缓解大模型幻觉", "expected": ["ref-010"]},
    {"query": "思维链提示 逐步推理", "expected": ["ref-011"]},
    {"query": "推理与行动结合 调用外部工具 智能体", "expected": ["ref-012"]},
    {"query": "思维树 树状搜索 回溯 复杂规划", "expected": ["ref-013"]},
    {"query": "自洽性 多条推理链采样投票", "expected": ["ref-014"]},
    {"query": "稠密段落检索 双编码器 开放域问答", "expected": ["ref-015"]},
    {"query": "LoRA 低秩适应 参数高效微调", "expected": ["ref-016"]},
    {"query": "FlashAttention 注意力加速 显存优化", "expected": ["ref-017"]},
    {"query": "词向量 分布式表示 神经网络 词嵌入", "expected": ["ref-018", "ref-021"]},
    {"query": "序列到序列 编码器解码器 机器翻译", "expected": ["ref-019", "ref-020"]},
    {"query": "残差网络 跳跃连接 深层网络退化", "expected": ["ref-025"]},
    {"query": "生成对抗网络 生成器与判别器 对抗训练", "expected": ["ref-026"]},
    {"query": "单阶段 实时目标检测", "expected": ["ref-027"]},
    {"query": "图文对比学习 零样本图像分类 多模态", "expected": ["ref-029"]},
    {"query": "潜在空间 扩散模型 高分辨率图像生成", "expected": ["ref-030", "ref-031"]},
    {"query": "图神经网络 图卷积 节点半监督分类", "expected": ["ref-037", "ref-038"]},
    {"query": "自适应学习率 一阶优化算法", "expected": ["ref-040"]},
    {"query": "联邦学习 隐私保护 分布式模型聚合", "expected": ["ref-041"]},
]


# ---------------------------------------------------------------------------
# 派生指标
# ---------------------------------------------------------------------------
def _derive_ranking(rows: List[Dict], k: int) -> Dict:
    """从明细行派生分级 Recall / MRR / 排名分布。"""
    ranks: List[Optional[int]] = []
    for r in rows:
        for eid in r["expected"]:
            ranks.append(r["retrieved"].index(eid) + 1 if eid in r["retrieved"] else None)

    total = len(ranks)

    def _recall_at(n: int) -> float:
        return round(sum(1 for x in ranks if x and x <= n) / total, 4) if total else None

    dist: Dict[str, int] = {}
    for x in ranks:
        if x:
            dist[str(x)] = dist.get(str(x), 0) + 1

    return {
        "recall_at_1": _recall_at(1),
        "recall_at_3": _recall_at(3),
        "recall_at_k": _recall_at(k),
        "mrr": round(sum(1.0 / x for x in ranks if x) / total, 4) if total else None,
        "rank_distribution": dict(sorted(dist.items(), key=lambda kv: int(kv[0]))),
        "unretrieved": [f"{r['query']} -> {eid}"
                        for r in rows for eid in r["expected"] if eid not in r["retrieved"]],
    }


def _tokens(s: str) -> set:
    """中英混排的粗粒度分词：中文按单字、英文/数字按词。"""
    s = str(s).lower()
    words = set(re.findall(r"[a-z][a-z0-9\-]+", s))
    zh = set(re.findall(r"[\u4e00-\u9fff]", s))
    return words | zh


def _attribution(rows: List[Dict]) -> Dict:
    """归因：查询词与「命中文献 / 非命中文献」的词汇重合度。

    若命中重合率远高于非命中，说明查询主要抄自目标文献原文，
    Recall 高是任务构造使然 —— 这是必须主动交代的前提。
    """
    from app.rag.corpus import ACADEMIC_REFERENCES

    def _full_text(d: Dict) -> str:
        meta = " ".join(str(v) for v in d["metadata"].values())
        return meta + " " + (d.get("text") or "")

    per_query = []
    for r in rows:
        q = _tokens(r["query"])
        if not q:
            continue
        hid = r["expected"][0]
        hit_doc = next((d for d in ACADEMIC_REFERENCES if d["id"] == hid), None)
        hit_ov = (len(q & _tokens(_full_text(hit_doc))) / len(q)) if hit_doc else 0.0
        others = [len(q & _tokens(_full_text(d))) / len(q)
                  for d in ACADEMIC_REFERENCES if d["id"] != hid]
        non_hit = sum(others) / len(others) if others else 0.0
        per_query.append({
            "query": r["query"],
            "expected": hid,
            "hit_overlap": round(hit_ov, 4),
            "non_hit_mean_overlap": round(non_hit, 4),
            "margin": round(hit_ov - non_hit, 4),
        })

    n = len(per_query)
    return {
        "mean_hit_overlap": round(sum(p["hit_overlap"] for p in per_query) / n, 4) if n else None,
        "mean_non_hit_overlap": round(sum(p["non_hit_mean_overlap"] for p in per_query) / n, 4) if n else None,
        "mean_margin": round(sum(p["margin"] for p in per_query) / n, 4) if n else None,
        "queries_with_full_overlap": sum(1 for p in per_query if p["hit_overlap"] >= 1.0),
        "per_query": per_query,
        "note": ("查询词取自目标文献原文，命中重合率≈97.6% 而语料内非命中仅≈14%，"
                 "故 Recall@K 偏高属构造性结果，只证明检索链路通畅，不代表泛化能力。"),
    }


def _enrich(rep: Dict) -> Dict:
    """给评测结果补上分级指标与归因（幂等，可重复调用）。"""
    rep["metrics"] = _derive_ranking(rep["rows"], rep["k"])
    rep["attribution"] = _attribution(rep["rows"])
    return rep


# ---------------------------------------------------------------------------
def _evaluate(top_k: int) -> Dict:
    corpus_size = vector_store.get_document_count()
    rows = []
    total_expected = 0
    total_hit = 0
    hit_at_k_queries = 0

    for case in EVAL_CASES:
        results = vector_store.search(case["query"], top_k=top_k)
        retrieved = [r["id"] for r in results]
        expected = case["expected"]
        hits = [ref for ref in expected if ref in retrieved]

        total_expected += len(expected)
        total_hit += len(hits)
        if hits:
            hit_at_k_queries += 1

        rows.append({
            "query": case["query"],
            "expected": expected,
            "retrieved": retrieved,
            "hits": hits,
            "missed": [ref for ref in expected if ref not in retrieved],
        })

    n = len(EVAL_CASES)
    return _enrich({
        "corpus_size": corpus_size,
        "k": top_k,
        "queries": n,
        "expected_total": total_expected,
        "hit_total": total_hit,
        "recall_at_k": round(total_hit / total_expected, 4) if total_expected else None,
        "hit_at_k": round(hit_at_k_queries / n, 4) if n else None,
        "rows": rows,
    })


def _print_report(rep: Dict) -> None:
    k = rep["k"]
    m = rep["metrics"]
    print(f"语料规模：{rep['corpus_size']} 篇    K={k}    查询数：{rep['queries']}")
    print("-" * 92)
    print(f"{'#':>3}  {'查询':<34}{'命中':>6}{'期望':>6}  最佳排名 / 命中文献")
    print("-" * 92)
    for i, r in enumerate(rep["rows"], 1):
        q = r["query"]
        q = q if len(q) <= 32 else q[:31] + "…"
        ranked = [r["retrieved"].index(e) + 1 if e in r["retrieved"] else "-"
                  for e in r["expected"]]
        mark = "OK" if not r["missed"] else "!!"
        print(f"{i:>3}  {q:<34}{len(r['hits']):>6}{len(r['expected']):>6}  "
              f"{mark} 排名{ranked} {','.join(r['hits']) or '-'}"
              + (f"  [漏: {','.join(r['missed'])}]" if r["missed"] else ""))
    print("-" * 92)
    print(f"Recall@1 = {m['recall_at_1'] * 100:.1f}%    "
          f"Recall@3 = {m['recall_at_3'] * 100:.1f}%    "
          f"Recall@{k} = {rep['recall_at_k'] * 100:.1f}%    MRR = {m['mrr']:.3f}")
    print(f"命中排名分布（排名:条数）= {m['rank_distribution']}")
    a = rep["attribution"]
    print(f"归因：命中文献词重合 {a['mean_hit_overlap'] * 100:.1f}% "
          f"vs 非命中 {a['mean_non_hit_overlap'] * 100:.1f}%"
          f"（区分度 {(a['mean_margin']) * 100:.1f}pp，"
          f"{a['queries_with_full_overlap']}/{rep['queries']} 条查询完全重合）")
    print(f"提示：{a['note']}")


def _derive_only(path: Path) -> int:
    """不调用向量库，仅从已有 JSON 的 rows 重算派生指标并回写。"""
    if not path.exists():
        print(f"[错误] 找不到已有结果：{path}", file=sys.stderr)
        return 2
    rep = json.loads(path.read_text(encoding="utf-8"))
    rep = _enrich(rep)
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(rep)
    print(f"\n派生指标已回写：{path}")
    return 0


def _main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索评测（分级 Recall + MRR + 归因）")
    parser.add_argument("--k", type=int, default=5, help="召回条数（默认 5）")
    parser.add_argument("--out", default="data/eval_recall.json", help="结果 JSON 输出路径")
    parser.add_argument("--derive-only", action="store_true",
                        help="不重跑检索，仅从已有 JSON 重算派生指标并回写")
    args = parser.parse_args()

    out = Path(args.out)
    if args.derive_only:
        return _derive_only(out)

    if vector_store.get_document_count() == 0:
        print("[错误] 向量库为空，请先执行 python seed_rag.py 灌入语料", file=sys.stderr)
        return 2

    rep = _evaluate(args.k)
    _print_report(rep)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

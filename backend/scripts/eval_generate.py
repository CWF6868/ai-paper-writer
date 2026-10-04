#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端生成评测 —— 真实调用 DeepSeek，逐篇跑完整工作流，产出三个量化指标：

    1. 平均生成时长（秒/篇，wall-clock）
    2. RAG 引用命中数（每篇参考文献条数 + 正文 [n] 引用标注数）
    3. 重写触发率（revision_count > 1 的论文占比；工作流里 quality_check 打分 < 0.7 会打回重写）

指标口径（对外报数请按此说明）：
    - revision_count = 质量检查节点的执行次数；首次检查不算重写，
      因此 revision_count >= 2 即「至少触发过一次重写」，1 表示一次通过。
    - 平均生成时长只统计 status=ok 的论文。
    - RAG 引用命中数 = 最终状态 state.references 的条数（检索命中 top_k=5 的文献）。

用法（在 backend/ 目录下执行，会真实消耗 API 额度）：
    python scripts/eval_generate.py                       # 跑全部选题
    python scripts/eval_generate.py --limit 2             # 只跑前 2 篇（先冒烟）
    python scripts/eval_generate.py --out data/eval_generate.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.llm import check_llm_config          # noqa: E402
from app.workflows.graph_builder import PaperWorkflowGraph  # noqa: E402

# ---------------------------------------------------------------------------
# 6 个选题：覆盖语料库里的主要方向（大模型 / RAG / 推理 / 检索 / 图 / 生成模型）
# ---------------------------------------------------------------------------
TOPICS: List[Dict] = [
    {"id": 1, "title": "基于检索增强生成的大语言模型幻觉缓解研究",
     "topic": "大语言模型幻觉缓解与检索增强生成", "keywords": "RAG,幻觉,大语言模型"},
    {"id": 2, "title": "大语言模型思维链推理能力增强方法研究",
     "topic": "大语言模型推理能力增强", "keywords": "思维链,推理,提示工程"},
    {"id": 3, "title": "面向开放域问答的稠密检索方法研究",
     "topic": "开放域问答与稠密向量检索", "keywords": "稠密检索,开放域问答,向量检索"},
    {"id": 4, "title": "图神经网络在知识图谱表示学习中的应用研究",
     "topic": "图神经网络与知识图谱表示学习", "keywords": "图神经网络,知识图谱,表示学习"},
    {"id": 5, "title": "面向中文大模型的参数高效微调方法研究",
     "topic": "大模型参数高效微调", "keywords": "LoRA,参数高效微调,大模型"},
    {"id": 6, "title": "扩散模型在文本到图像生成中的应用与优化研究",
     "topic": "扩散模型与文本到图像生成", "keywords": "扩散模型,文生图,图像生成"},
]

CITATION_RE = re.compile(r"\[(\d{1,3})\]")


def _word_count(text: str) -> int:
    """中文字数（去掉空白字符后的字符数）"""
    return len(re.sub(r"\s", "", text or ""))


async def _run_once(graph: PaperWorkflowGraph, item: Dict) -> Dict:
    state = {
        "paper_id": item["id"],
        "title": item["title"],
        "topic": item["topic"],
        "keywords": item["keywords"],
        "is_topic_clear": True,
        "revision_count": 0,
        "max_revisions": 2,
        "sections_content": {},
    }
    # 每篇一个独立线程 id，避免 MemorySaver 里不同论文互相串状态
    config = {"configurable": {"thread_id": f"eval-{item['id']}"}}

    t0 = time.perf_counter()
    final = await graph.run(state, config=config)
    seconds = time.perf_counter() - t0

    paper = final.get("final_paper") or ""
    refs = final.get("references") or []
    markers = sorted({int(m.group(1)) for m in CITATION_RE.finditer(paper)})

    return {
        "id": item["id"],
        "title": item["title"],
        "status": "ok" if paper else "empty",
        "seconds": round(seconds, 1),
        "word_count": _word_count(paper),
        "revision_count": final.get("revision_count", 0),
        "rewrite_triggered": final.get("revision_count", 0) > 1,
        "quality_score": final.get("quality_score"),
        "sections": len(final.get("sections_content") or {}),
        "references_count": len(refs),
        "reference_titles": [r.get("title") for r in refs],
        "body_citation_markers": markers,
        "body_citation_count": len(markers),
    }


def _summarize(rows: List[Dict]) -> Dict:
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return {"n": 0}

    secs = [r["seconds"] for r in ok]
    revisions = [r["revision_count"] for r in ok]
    triggered = [r for r in ok if r["rewrite_triggered"]]

    return {
        "n": len(ok),
        "total_seconds": round(sum(secs), 1),
        "avg_seconds": round(statistics.mean(secs), 1),
        "min_seconds": round(min(secs), 1),
        "max_seconds": round(max(secs), 1),
        "median_seconds": round(statistics.median(secs), 1),
        "avg_word_count": round(statistics.mean([r["word_count"] for r in ok])),
        "avg_references": round(statistics.mean([r["references_count"] for r in ok]), 2),
        "total_references": sum(r["references_count"] for r in ok),
        "avg_body_citations": round(statistics.mean([r["body_citation_count"] for r in ok]), 2),
        "rewrite_triggered_count": len(triggered),
        "rewrite_trigger_rate": round(len(triggered) / len(ok), 4),
        "avg_revision_count": round(statistics.mean(revisions), 2),
        "revision_count_distribution": {str(k): revisions.count(k) for k in sorted(set(revisions))},
        # revision_count 记的是「质量检查执行次数」；重写轮次 = 检查次数 - 1（首次检查不算重写）
        "avg_rewrite_rounds": round(statistics.mean([max(0, r - 1) for r in revisions]), 2),
        "rewrite_rounds_distribution": {
            str(k): [max(0, r - 1) for r in revisions].count(k)
            for k in sorted(set(max(0, r - 1) for r in revisions))
        },
        "avg_quality_score": round(statistics.mean(
            [r["quality_score"] for r in ok if r["quality_score"] is not None] or [0]), 4),
    }


async def _main() -> int:
    parser = argparse.ArgumentParser(description="端到端论文生成评测（真实调用 DeepSeek）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 篇（0 = 全部）")
    parser.add_argument("--out", default="data/eval_generate.json", help="结果 JSON 输出路径")
    args = parser.parse_args()

    check_llm_config()                      # 缺 Key 立即失败，别跑到一半才报错

    topics = TOPICS[: args.limit] if args.limit else TOPICS
    graph = PaperWorkflowGraph()

    rows: List[Dict] = []
    for i, item in enumerate(topics, 1):
        print(f"[{i}/{len(topics)}] 开始生成：{item['title']}", flush=True)
        try:
            row = await _run_once(graph, item)
        except Exception as e:              # 单篇失败不影响其余论文
            row = {"id": item["id"], "title": item["title"], "status": "error", "error": repr(e)}
        rows.append(row)
        if row["status"] == "ok":
            print(f"    -> 完成 {row['seconds']}s | {row['word_count']} 字 | "
                  f"文献 {row['references_count']} 条 | 正文引用 {row['body_citation_count']} 处 | "
                  f"重写次数 {row['revision_count']} | 质量分 {row['quality_score']}", flush=True)
        else:
            print(f"    -> 失败：{row.get('error')}", flush=True)

    summary = _summarize(rows)
    report = {"summary": summary, "papers": rows}

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    if summary.get("n"):
        print(f"  成功篇数          : {summary['n']}")
        print(f"  平均生成时长      : {summary['avg_seconds']}s"
              f"（最短 {summary['min_seconds']}s / 最长 {summary['max_seconds']}s / 中位 {summary['median_seconds']}s）")
        print(f"  平均字数          : {summary['avg_word_count']} 字")
        print(f"  RAG 引用命中数    : 平均 {summary['avg_references']} 条/篇，合计 {summary['total_references']} 条")
        print(f"  正文 [n] 引用     : 平均 {summary['avg_body_citations']} 处/篇")
        print(f"  重写触发率        : {summary['rewrite_triggered_count']}/{summary['n']} = "
              f"{summary['rewrite_trigger_rate'] * 100:.1f}%"
              f"（平均重写轮次 {summary['avg_rewrite_rounds']}，质检平均执行 "
              f"{summary['avg_revision_count']} 次）")
        print(f"  重写轮次分布      : {summary['rewrite_rounds_distribution']}")
        print(f"  平均质量分        : {summary['avg_quality_score']}")
    else:
        print("  没有任何一篇成功，请检查 API Key / 网络")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""并发吞吐基准 —— 用固定延迟的 stub Agent 替换 4 个 LLM Agent，测工作流架构本身的并发能力。

为什么不直接压真实 API：
    1. DeepSeek 的响应时延抖动（±30%）会淹没架构差异，测 3 次得 3 个数；
    2. 「N 篇并发 × K 章并发」= N×K 个在途请求，测出来的是供应商的限额，不是我们的代码；
    3. 真实调用要花额度，没法每改一行就重跑。

所以这里把 LLM 调用换成 `await asyncio.sleep(固定延迟)`，**前后两次跑用的是同一份延迟参数**，
差异只可能来自「章节写作是否并发」。这是可控对比，代价是绝对耗时不是真实耗时（只做相对比较）。

口径（对外报数请按此说明）：
    - 每篇论文 sections 章（默认 8），writer 每次调用固定 writer_latency 秒（默认 1.0）；
    - 吞吐 QPS = 完成篇数 / 墙钟分钟（篇/分钟，"单机并发 QPS" 即此项）；
    - --levels 是被测的「论文级并发数」，每一级单独计时（不同级之间串行跑，不叠加）。

用法（backend/ 目录下）：
    python scripts/bench_concurrency.py --tag before
    python scripts/bench_concurrency.py --tag after
    python scripts/bench_concurrency.py --levels 1,2,4,8 --papers 8 --writer-latency 2.0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.workflows.graph_builder import PaperWorkflowGraph  # noqa: E402

# 各节点的固定延迟（秒）。改前改后必须完全一致，否则对比无意义。
DEFAULT_LATENCY = {
    "outline": 0.8,
    "retrieve": 0.1,
    "polish": 1.0,
    "review": 1.2,
}


class _StubAgent:
    """把 Agent 的 execute 换成一个固定延迟的协程，返回值形状与真 Agent 一致。"""

    def __init__(self, latency: float, build: Callable[[dict], dict]):
        self.latency = latency
        self.build = build
        self.calls = 0
        self.in_flight = 0
        self.peak_in_flight = 0

    async def execute(self, payload: dict) -> dict:
        self.calls += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.latency)
            return self.build(payload)
        finally:
            self.in_flight -= 1


def _install_stubs(graph: PaperWorkflowGraph, sections: int, writer_latency: float) -> Dict[str, _StubAgent]:
    """给图上的 4 个 Agent 换上 stub；返回 stub 表（用于统计调用次数/峰值并发）。"""
    outline_body = {
        "sections": [
            {"title": f"第{i}章 测试章节", "points": ["要点A", "要点B"], "word_count": 500}
            for i in range(1, sections + 1)
        ]
    }

    stubs = {
        "outline": _StubAgent(DEFAULT_LATENCY["outline"], lambda _p: {"outline_dict": outline_body}),
        "writer": _StubAgent(writer_latency, lambda p: {"content": f"《{p.get('section_title', '')}》正文"}),
        "polish": _StubAgent(DEFAULT_LATENCY["polish"], lambda _p: {"polished": "润色后的完整论文"}),
        "review": _StubAgent(
            DEFAULT_LATENCY["review"],
            # 0.9 > 阈值 → 一轮通过，不触发重写：让每篇的调用序列固定，耗时可推演
            lambda _p: {"raw": "{}", "data": {"score": 0.9, "overall": "好", "dimensions": []}},
        ),
    }

    graph.outline_agent.execute = stubs["outline"].execute
    graph.writer_agent.execute = stubs["writer"].execute
    graph.polish_agent.execute = stubs["polish"].execute
    graph.reviewer_agent.execute = stubs["review"].execute

    fake_refs = [
        {"id": f"ref-{i}", "title": f"文献{i}", "authors": "张三", "journal": "测试刊", "year": 2020}
        for i in range(1, 6)
    ]

    async def fake_retrieve(query: str, top_k: int = 5):
        await asyncio.sleep(DEFAULT_LATENCY["retrieve"])
        return fake_refs

    graph.reference_agent.retrieve = fake_retrieve
    return stubs


async def _run_paper(graph: PaperWorkflowGraph, idx: int) -> float:
    state = {
        "paper_id": idx,
        "title": f"基准论文 {idx}",
        "topic": "并发基准",
        "keywords": ["bench"],
        "is_topic_clear": True,
        "revision_count": 0,
        "max_revisions": 2,
        "sections_content": {},
    }
    t0 = time.perf_counter()
    await graph.run(state, config={"configurable": {"thread_id": f"bench-{idx}"}})
    return time.perf_counter() - t0


async def _run_level(graph: PaperWorkflowGraph, level: int, papers: int) -> Dict:
    """论文级并发 = level：同时在跑 level 篇，统计整体吞吐。"""
    sem = asyncio.Semaphore(level)
    durations: List[float] = []

    async def _one(i: int):
        async with sem:
            d = await _run_paper(graph, i)
            durations.append(d)

    t0 = time.perf_counter()
    await asyncio.gather(*[_one(i) for i in range(papers)])
    elapsed = time.perf_counter() - t0

    return {
        "paper_concurrency": level,
        "papers": papers,
        "elapsed_seconds": round(elapsed, 2),
        "avg_paper_seconds": round(statistics.mean(durations), 2),
        "p50_paper_seconds": round(statistics.median(durations), 2),
        # 吞吐：篇/分钟。这就是「单机并发 QPS」的口径（QPS 按业务次数=篇 计）
        "throughput_per_min": round(papers / (elapsed / 60), 3),
    }


async def _main() -> int:
    parser = argparse.ArgumentParser(description="并发吞吐基准（stub LLM，不消耗 API 额度）")
    parser.add_argument("--papers", type=int, default=6, help="每一级并发跑多少篇")
    parser.add_argument("--levels", default="1,2,4", help="论文级并发数，逗号分隔")
    parser.add_argument("--sections", type=int, default=8, help="每篇章节数")
    parser.add_argument("--writer-latency", type=float, default=1.0, help="单次章节写作的模拟耗时（秒）")
    parser.add_argument("--tag", default="run", help="标记本次结果（before / after）")
    parser.add_argument("--out", default="data/bench_concurrency.json", help="结果输出路径")
    args = parser.parse_args()

    levels = [int(x) for x in str(args.levels).split(",") if x.strip()]
    graph = PaperWorkflowGraph()
    stubs = _install_stubs(graph, args.sections, args.writer_latency)

    print(f"参数：papers={args.papers} sections={args.sections} "
          f"writer_latency={args.writer_latency}s levels={levels}", flush=True)

    rows = []
    for lv in levels:
        for s in stubs.values():          # 每级重置计数
            s.calls = 0
            s.peak_in_flight = 0
        row = await _run_level(graph, lv, args.papers)
        row["writer_peak_in_flight"] = stubs["writer"].peak_in_flight
        row["writer_calls"] = stubs["writer"].calls
        rows.append(row)
        print(f"  论文级并发 {lv:>2} → 总耗时 {row['elapsed_seconds']:>6.2f}s | "
              f"单篇均值 {row['avg_paper_seconds']:>5.2f}s | "
              f"吞吐 {row['throughput_per_min']:>6.2f} 篇/分钟 | "
              f"写作峰值在途 {row['writer_peak_in_flight']}", flush=True)

    base = rows[0]["throughput_per_min"] if rows else 0
    for r in rows:
        r["speedup_vs_first"] = round(r["throughput_per_min"] / base, 2) if base else None

    report = {
        "tag": args.tag,
        "params": {
            "papers": args.papers,
            "sections": args.sections,
            "writer_latency": args.writer_latency,
            "levels": levels,
            "node_latency": DEFAULT_LATENCY,
        },
        "results": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    if isinstance(existing, dict) and "runs" in existing:
        existing["runs"][args.tag] = report
        payload = existing
    else:
        payload = {"runs": {args.tag: report}}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入：{out}（tag={args.tag}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

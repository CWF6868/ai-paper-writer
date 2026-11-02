#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比两次 eval_generate 结果，输出可直接粘贴进文档的 Markdown 表。

为什么需要它：
    两次实跑的「重写轮次」可能不同（阈值附近抖动就会导致一篇少写一轮），
    直接比总耗时是不公平的 —— 少一轮重写等于少写 8 章。
    所以除了总耗时，这里还输出 **每章次耗时 = 总耗时 / (章节数 × 质检次数)**，
    把「章节数」和「重写轮次」两个变量都归一化掉，这才是能横向比的数字。

用法（backend/ 目录下）：
    python scripts/compare_eval.py --before data/eval_generate_v4.json --after data/eval_generate_v5.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List


def _load(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pct(before: float, after: float) -> str:
    if not before:
        return "—"
    return f"{(after - before) / before * 100:+.1f}%"


def _fmt(v, digits=1):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _per_section_write(row: Dict) -> float | None:
    """每章次耗时：总耗时 / (章节数 × 质检次数)。

    质检次数 = revision_count，而每轮质检前都会把全部章节重写一遍，
    所以这个分母等于"实际发生的章节写作次数"，与章节数、重写轮次都解耦。
    """
    sections = row.get("sections") or 0
    rounds = row.get("revision_count") or 0
    if not sections or not rounds:
        return None
    return row["seconds"] / (sections * rounds)


def main() -> int:
    ap = argparse.ArgumentParser(description="对比两份 eval_generate 结果")
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--labels", default="改造前,改造后")
    args = ap.parse_args()

    b, a = _load(args.before), _load(args.after)
    lb, la = args.labels.split(",")
    sb, sa = b.get("summary", {}), a.get("summary", {})

    print(f"## 端到端对比：{lb} vs {la}\n")
    print("| 指标 | {} | {} | 变化 |".format(lb, la))
    print("|---|---|---|---|")

    rows = [
        ("成功篇数", "n", 0, ""),
        ("平均耗时 (s/篇)", "avg_seconds", 1, "lower"),
        ("中位耗时 (s)", "median_seconds", 1, "lower"),
        ("合计耗时 (s)", "total_seconds", 1, "lower"),
        ("平均字数", "avg_word_count", 0, ""),
        ("平均质量分", "avg_quality_score", 3, ""),
        ("重写触发率", "rewrite_trigger_rate", 4, ""),
        ("RAG 引用命中 (条/篇)", "avg_references", 2, ""),
        ("正文引用标注 (处/篇)", "avg_body_citations", 2, ""),
    ]
    for label, key, digits, _dir in rows:
        vb, va = sb.get(key), sa.get(key)
        if isinstance(vb, float) and key in ("rewrite_trigger_rate",):
            cell_b, cell_a = f"{vb * 100:.0f}%", f"{va * 100:.0f}%"
            delta = f"{(va - vb) * 100:+.0f}pp" if va is not None and vb is not None else "—"
        else:
            cell_b, cell_a = _fmt(vb, digits), _fmt(va, digits)
            # 篇数是计数，百分比没有意义（6 → 6 写成 +0.0% 只会误导）
            delta = "—" if key == "n" else (_pct(vb, va) if isinstance(vb, (int, float)) and vb else "—")
        print(f"| {label} | {cell_b} | {cell_a} | {delta} |")

    # 归一化指标
    pb = [x for x in (_per_section_write(r) for r in b.get("papers", []) if r.get("status") == "ok") if x]
    pa = [x for x in (_per_section_write(r) for r in a.get("papers", []) if r.get("status") == "ok") if x]
    if pb and pa:
        mb, ma = statistics.mean(pb), statistics.mean(pa)
        print(f"| **每章次耗时 (s)** | {mb:.1f} | {ma:.1f} | **{_pct(mb, ma)}** |")

    print("\n### 逐篇明细\n")
    print(f"| id | {lb} 耗时 | {la} 耗时 | 变化 | {lb} 章×轮 | {la} 章×轮 | {lb} 质量 | {la} 质量 |")
    print("|---|---|---|---|---|---|---|---|")
    bmap = {r.get("id"): r for r in b.get("papers", [])}
    for row in a.get("papers", []):
        old = bmap.get(row.get("id"), {})
        if row.get("status") != "ok":
            print(f"| {row.get('id')} | — | 失败 | — | — | — | — | — |")
            continue
        secs_b, secs_a = old.get("seconds"), row.get("seconds")
        print("| {} | {} | {} | {} | {}×{} | {}×{} | {} | {} |".format(
            row.get("id"),
            _fmt(secs_b), _fmt(secs_a),
            _pct(secs_b, secs_a) if secs_b else "—",
            old.get("sections", "—"), old.get("revision_count", "—"),
            row.get("sections", "—"), row.get("revision_count", "—"),
            _fmt(old.get("quality_score"), 3), _fmt(row.get("quality_score"), 3),
        ))

    # 质量漂移提示：分数/字数的变化是否超出噪声带
    qb = [r.get("quality_score") for r in b.get("papers", []) if r.get("quality_score") is not None]
    qa = [r.get("quality_score") for r in a.get("papers", []) if r.get("quality_score") is not None]
    if qb and qa:
        dq = statistics.mean(qa) - statistics.mean(qb)
        wb, wa = sb.get("avg_word_count") or 0, sa.get("avg_word_count") or 0
        dw = (wa - wb) / wb * 100 if wb else 0
        print("\n### 质量漂移检查\n")
        print(f"- 质量分变化 {dq:+.3f}（噪声带 ±0.02）：{'在带内' if abs(dq) <= 0.02 else '**超出噪声带，需解释**'}")
        print(f"- 平均字数变化 {dw:+.1f}%（阈值 ±10%）：{'在带内' if abs(dw) <= 10 else '**超出阈值，需解释**'}")
        pb2 = set().union(*[set(r.get("body_citation_markers") or []) for r in b.get("papers", [])])
        pa2 = set().union(*[set(r.get("body_citation_markers") or []) for r in a.get("papers", [])])
        print(f"- 正文引用编号出现范围：{lb} {sorted(pb2)} → {la} {sorted(pa2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

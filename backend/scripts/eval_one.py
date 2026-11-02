#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单篇重跑 —— 评测里个别篇目失败/降级时，只补跑那一篇，避免整轮重跑浪费额度。

与 eval_generate.py 共用同一套 _run_once / TOPICS 口径，保证补跑结果
与整轮结果可合并比较（字段完全一致）。

用法（在 backend/ 目录下执行，会真实消耗 API 额度）：
    python scripts/eval_one.py 2                                  # 补跑 id=2
    python scripts/eval_one.py 2 5                                # 补跑 2 和 5
    python scripts/eval_one.py 2 --out data/eval_patch_2.json     # 指定输出
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.utils.llm import check_llm_config                      # noqa: E402
from app.workflows.graph_builder import PaperWorkflowGraph      # noqa: E402
from eval_generate import TOPICS, _run_once                     # noqa: E402


async def main_async(ids, out: Path) -> int:
    picked = [t for t in TOPICS if t["id"] in ids]
    missing = set(ids) - {t["id"] for t in picked}
    if missing:
        print(f"未知 id：{sorted(missing)}（可选 {[t['id'] for t in TOPICS]}）")
        return 1

    graph = PaperWorkflowGraph()
    rows = []
    for item in picked:
        print(f"[补跑 {item['id']}] {item['title']}", flush=True)
        try:
            row = await _run_once(graph, item)
        except Exception as e:                      # 单篇失败不影响其余
            row = {"id": item["id"], "title": item["title"], "status": "error", "error": str(e)}
        flag = "" if row.get("quality_score") is not None else "  ← 降级（质量分为空）"
        print(f"    -> {row['status']} | {row.get('seconds')}s | 字数 {row.get('word_count')}"
              f" | 首检 {row.get('first_pass_score')} | 最终 {row.get('quality_score')}{flag}",
              flush=True)
        rows.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"papers": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入：{out}")
    ok = [r for r in rows if r.get("status") == "ok" and r.get("quality_score") is not None]
    print(f"有效样本 {len(ok)}/{len(rows)} 篇（可用于阈值标定）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", type=int)
    ap.add_argument("--out", default="data/eval_patch.json")
    args = ap.parse_args()
    check_llm_config()
    return asyncio.run(main_async(set(args.ids), Path(args.out)))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""引用一致性校验脚本 —— 交叉核对「正文 [n] 标注」与「文末参考文献列表」。

为什么要有这个脚本（口径问题）：
    直接按行切开模型输出、把每一行都当「文献条目」，会把「以下按 GB/T 7714 格式整理……」
    这类说明性文字算进分母，于是出现 total=9 / matched=5 → 55.6% 这种失真数字。
    本脚本固定口径：**只有形如 `[n] 内容` 的行才算真实条目**，其余一律不计入分母。

三条可复现的指标：
    1. 正文引用覆盖率 = 正文用到的编号里，能在文末列表找到对应条目的比例
       （对不上的是「悬空引用」——正文引了、列表没有，疑似编造）
    2. 条目被引用率   = 文末条目里，正文真正引用过的比例
       （没被引用的是「白列条目」——列了却没在正文用上）
    3. 条目可溯源率   = 文末条目的标题能在固定语料库（app/rag/corpus.py，42 篇）中找到的比例

用法（在 backend/ 目录下执行）：
    python scripts/verify_citations.py ../../output_paper.md          # 单篇 Markdown
    python scripts/verify_citations.py data/eval_generate.json --eval  # 批量校验评测结果
    python scripts/verify_citations.py xxx.md --json                   # 机器可读输出
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 正文 / 列表里的引用编号： [1] [12]
CITATION_RE = re.compile(r"\[(\d{1,3})\]")
# 真实参考文献条目：行首为 [n] 且后面有内容
ENTRY_RE = re.compile(r"^\s*\[(\d{1,3})\]\s*(\S.*?)\s*$")
# 参考文献标题行（支持 "## 参考文献" / "参考文献" / "References"）
REF_HEADING_RE = re.compile(r"^\s*#{0,6}\s*(参考文献|References)\s*[:：]?\s*$", re.MULTILINE | re.IGNORECASE)
# 代码块（避免把示例代码里的 [1] 计入统计）
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def split_paper(text: str) -> tuple:
    """把论文切成 (正文, 参考文献部分)。

    优先按「参考文献」标题切；没有标题时，退化为「最后一个以 [n] 开头的连续行块」。
    """
    m = REF_HEADING_RE.search(text)
    if m:
        return text[: m.start()], text[m.end():]

    lines = text.splitlines()
    idx = [i for i, ln in enumerate(lines) if ENTRY_RE.match(ln)]
    if not idx:
        return text, ""

    start = idx[0]
    for prev, cur in zip(idx, idx[1:]):
        if cur - prev > 3:      # 中间断开较远，说明前面那段不是参考文献列表
            start = cur
    return "\n".join(lines[:start]), "\n".join(lines[start:])


def parse_entries(ref_text: str) -> Dict[int, str]:
    """解析参考文献列表，只保留形如 `[n] 内容` 的真实条目"""
    entries: Dict[int, str] = {}
    for line in ref_text.splitlines():
        m = ENTRY_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        entries.setdefault(idx, m.group(2))   # 编号重复时保留首次出现
    return entries


def extract_markers(body_text: str) -> List[int]:
    """提取正文里用到的引用编号（去重、升序；忽略代码块）"""
    body = FENCE_RE.sub("", body_text)
    return sorted({int(m.group(1)) for m in CITATION_RE.finditer(body)})


def _corpus_titles() -> set:
    try:
        from app.rag.corpus import ACADEMIC_REFERENCES
        return {d["metadata"]["title"] for d in ACADEMIC_REFERENCES}
    except Exception:
        return set()


def _fmt_rate(rate: Optional[float]) -> str:
    return "N/A" if rate is None else f"{rate * 100:.1f}%"


def verify_paper(text: str, name: str = "<paper>") -> Dict:
    """单篇交叉核对：正文引用 ↔ 文末条目"""
    body, refs = split_paper(text)
    entries = parse_entries(refs)
    markers = extract_markers(body)

    matched = [n for n in markers if n in entries]
    dangling = [n for n in markers if n not in entries]     # 悬空引用（疑似编造）
    uncited = sorted(set(entries) - set(markers))           # 白列条目

    corpus = _corpus_titles()
    traceable = [
        idx for idx, title in entries.items()
        if any(t and t in title for t in corpus)            # 标题包含语料库标题即可判定可溯源
    ]

    return {
        "name": name,
        "markers_total": len(markers),
        "entries_total": len(entries),
        "matched": len(matched),
        "dangling": dangling,
        "uncited": uncited,
        "citation_coverage": (len(matched) / len(markers)) if markers else None,
        "entries_cited_rate": ((len(entries) - len(uncited)) / len(entries)) if entries else None,
        "traceable_entries": len(traceable),
        "entry_traceability": (len(traceable) / len(entries)) if entries else None,
    }


def verify_eval_results(path: Path) -> Dict:
    """批量校验 ev生成评测结果 JSON：只用到 references 标题与正文引用编号"""
    data = json.loads(path.read_text(encoding="utf-8"))
    papers = data.get("papers") if isinstance(data, dict) else data
    corpus = _corpus_titles()

    rows = []
    for rec in papers or []:
        if not isinstance(rec, dict):
            continue
        titles = rec.get("reference_titles") or []
        markers = rec.get("body_citation_markers") or []
        traceable = [t for t in titles if t in corpus]
        rows.append({
            "paper": rec.get("title"),
            "status": rec.get("status"),
            "entries": len(titles),
            "entries_traceable": len(traceable),
            "markers": markers,
            "markers_consistent": bool(markers) and set(markers) <= set(range(1, len(titles) + 1)),
        })
    return {"source": str(path), "rows": rows}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="引用一致性校验（正文 [n] ↔ 参考文献列表）")
    parser.add_argument("path", nargs="?", default="../../output_paper.md", help="论文 Markdown 或评测结果 JSON")
    parser.add_argument("--eval", action="store_true", help="把 path 当作生成评测结果 JSON 批量校验")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非文本表格")
    args = parser.parse_args(argv)

    target = Path(args.path)
    if not target.exists():
        print(f"[错误] 文件不存在：{target}", file=sys.stderr)
        return 2

    if args.eval:
        report = verify_eval_results(target)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0

        rows = report["rows"]
        print(f"评测结果：{target}    论文数：{len(rows)}")
        print("-" * 96)
        print(f"{'论文':<34}{'条目':>6}{'可溯源':>8}{'正文编号':>10}{'编号一致':>10}")
        print("-" * 96)
        for r in rows:
            title = str(r["paper"] or "")
            title = title if len(title) <= 32 else title[:31] + "…"
            print(f"{title:<34}{r['entries']:>6}{r['entries_traceable']:>8}"
                  f"{str(r['markers']):>10}{('是' if r['markers_consistent'] else '否'):>10}")
        print("-" * 96)

        entries = sum(r["entries"] for r in rows)
        traceable = sum(r["entries_traceable"] for r in rows)
        consistent = sum(1 for r in rows if r["markers_consistent"])
        print(f"合计：条目可溯源 {traceable}/{entries} = {_fmt_rate(traceable / entries if entries else None)}；"
              f"正文编号与列表一致 {consistent}/{len(rows)} 篇")
        return 0

    text = target.read_text(encoding="utf-8")
    rep = verify_paper(text, name=target.name)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0

    print(f"论文：{rep['name']}")
    print(f"  正文引用编号数：{rep['markers_total']}    参考文献条目数：{rep['entries_total']}")
    print(f"  引用覆盖率（正文编号能在列表找到）：{rep['matched']}/{rep['markers_total']}"
          f" = {_fmt_rate(rep['citation_coverage'])}")
    print(f"  条目被引用率（列表条目被正文引用）：{rep['entries_total'] - len(rep['uncited'])}/{rep['entries_total']}"
          f" = {_fmt_rate(rep['entries_cited_rate'])}")
    print(f"  条目可溯源率（标题命中 42 篇语料）：{rep['traceable_entries']}/{rep['entries_total']}"
          f" = {_fmt_rate(rep['entry_traceability'])}")
    if rep["dangling"]:
        print(f"  [警告] 悬空引用（正文引了但列表没有，疑似编造）：{rep['dangling']}")
    if rep["uncited"]:
        print(f"  [提示] 白列条目（列表里有但正文没引）：{rep['uncited']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

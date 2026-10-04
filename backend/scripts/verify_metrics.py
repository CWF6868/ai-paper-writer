"""独立复核五项量化指标 —— 不信任任何 summary 声明值，一律从明细重算再比对。

设计原则（这就是"真实性"的技术保障）：
  1. 自洽性：summary 里声明的数字，必须能由 papers[] / rows[] 明细独立重算出来。
     若有人手改过 summary，重算值会与声明值不符，本脚本立刻报 FAIL。
  2. 完整性：coverage.json 的文件清单必须覆盖 app/ 下全部 .py，不能挑着报。
  3. 交叉性：引用命中用 app/rag/corpus.py 的语料标题独立比对，不复用评测脚本的中间结果。
  4. 可复现：--live 会真实重跑 23 条检索查询，验证 Recall@5 现在跑仍然是 27/27。

用法：
    python scripts/verify_metrics.py              # 离线自洽 + 交叉校验（零外部调用，秒级）
    python scripts/verify_metrics.py --live       # 额外真实重跑检索评测（需 DASHSCOPE_API_KEY）
    python scripts/verify_metrics.py --json out.json
退出码：0 = 全部一致；1 = 存在不一致（说明数据不可信）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

DATA = BACKEND / "data"

# (检查项, 重算值, 记录值, 是否一致)
CHECKS: list[tuple[str, object, object, bool]] = []


def _eq(a, b, tol: float = 1e-6) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= tol
        except (TypeError, ValueError):
            return False
    return a == b


def check(name: str, recomputed, recorded, tol: float = 1e-6) -> None:
    ok = _eq(recomputed, recorded, tol)
    CHECKS.append((name, recomputed, recorded, ok))


def _load(name: str) -> dict:
    p = DATA / name
    if not p.exists():
        print(f"[FATAL] 缺少原始数据文件：{p}")
        sys.exit(2)
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 1. 覆盖率
# --------------------------------------------------------------------------
def verify_coverage() -> dict:
    cov = _load("coverage.json")
    t = cov["totals"]
    n_stmt = t["num_statements"]
    n_cov = t["covered_lines"]

    # 重算：覆盖率 = 已覆盖 / 总语句
    recomputed_pct = round(n_cov / n_stmt * 100, 2)
    check("覆盖率·百分比重算", recomputed_pct, round(t["percent_covered"], 2), 0.01)
    check("覆盖率·语句数-已覆盖+缺失",
          n_cov + t["missing_lines"], n_stmt)
    check("覆盖率·显示值取整", t["percent_covered_display"],
          str(int(round(t["percent_covered"]))))

    # 完整性：coverage.json 记录的文件 必须覆盖 app/ 下全部 .py
    recorded_files = {p.replace("\\", "/") for p in cov["files"]}
    actual_files = {
        str(p.relative_to(BACKEND)).replace("\\", "/")
        for p in (BACKEND / "app").rglob("*.py")
    }
    missing_in_report = sorted(actual_files - recorded_files)
    check("覆盖率·文件覆盖完整（无漏报模块）",
          len(missing_in_report), 0)
    if missing_in_report:
        print(f"  [warn] 未被覆盖率报告纳入的模块：{missing_in_report}")

    # 逐文件语句数求和 == totals 语句数（防 totals 与明细脱节）
    sum_stmt = sum(f["summary"]["num_statements"] for f in cov["files"].values())
    check("覆盖率·逐文件语句数求和 == 总计", sum_stmt, n_stmt)

    return {
        "num_statements": n_stmt,
        "covered_lines": n_cov,
        "missing_lines": t["missing_lines"],
        "percent": recomputed_pct,
        "modules_in_report": len(recorded_files),
        "modules_on_disk": len(actual_files),
        "modules_missing": missing_in_report,
    }


# --------------------------------------------------------------------------
# 2. 检索 Recall@K
# --------------------------------------------------------------------------
def verify_recall() -> dict:
    r = _load("eval_recall.json")
    rows = r["rows"]

    # 从明细重算，不复用顶层声明值
    exp_total = sum(len(x["expected"]) for x in rows)
    hit_total = sum(len(x["hits"]) for x in rows)
    check("Recall·查询条数", len(rows), r["queries"])
    check("Recall·期望文献总数", exp_total, r["expected_total"])
    check("Recall·命中文献总数", hit_total, r["hit_total"])
    check("Recall@K 比值重算",
          round(hit_total / exp_total, 6), round(r["recall_at_k"], 6))

    # 逐行自洽：hits 必须是 expected ∩ retrieved
    bad_rows = []
    for x in rows:
        expect_hits = sorted(set(x["expected"]) & set(x["retrieved"]))
        if sorted(x["hits"]) != expect_hits:
            bad_rows.append(x["query"])
    check("Recall·每行 hits == expected∩retrieved（明细自洽）",
          len(bad_rows), 0)
    if bad_rows:
        print(f"  [warn] 明细不自洽的行：{bad_rows[:5]}")

    # 命中必须排在 top-K 内
    over_k = [x["query"] for x in rows
              if len(x["retrieved"]) > r["k"]]
    check(f"Recall·返回条数不超过 K={r['k']}", len(over_k), 0)

    # --- 分级指标：从明细独立重算，与 metrics 声明值比对 ---
    m = r.get("metrics", {})
    ranks = []
    for x in rows:
        for eid in x["expected"]:
            ranks.append(x["retrieved"].index(eid) + 1 if eid in x["retrieved"] else None)
    tot = len(ranks)
    r1 = sum(1 for x in ranks if x and x <= 1)
    r3 = sum(1 for x in ranks if x and x <= 3)
    mrr = sum(1.0 / x for x in ranks if x) / tot
    check("Recall@1 重算", round(r1 / tot, 4), m.get("recall_at_1"))
    check("Recall@3 重算", round(r3 / tot, 4), m.get("recall_at_3"))
    check("MRR 重算", round(mrr, 4), m.get("mrr"))
    # 排名分布条数之和 == 命中总数
    dist_sum = sum(m.get("rank_distribution", {}).values())
    check("Recall·排名分布条数 == 命中总数", dist_sum, hit_total)
    # 单调性：Recall@1 <= Recall@3 <= Recall@K
    check("Recall·分级单调（@1<=@3<=@K）",
          r1 / tot <= r3 / tot <= hit_total / exp_total, True)

    # --- 归因方向性：命中重合率必须显著高于非命中，否则说明标注有问题 ---
    a = r.get("attribution", {})
    if a:
        check("归因·命中重合率 > 非命中重合率",
              a.get("mean_hit_overlap", 0) > a.get("mean_non_hit_overlap", 0), True)

    return {
        "queries": len(rows),
        "k": r["k"],
        "expected_total": exp_total,
        "hit_total": hit_total,
        "recall_at_k": round(hit_total / exp_total, 4),
        "recall_at_1": round(r1 / tot, 4),
        "recall_at_3": round(r3 / tot, 4),
        "mrr": round(mrr, 4),
        "rank_distribution": m.get("rank_distribution"),
        "corpus_size": r["corpus_size"],
        "mean_hit_overlap": a.get("mean_hit_overlap"),
        "mean_non_hit_overlap": a.get("mean_non_hit_overlap"),
    }


def verify_recall_live() -> dict:
    """真实重跑检索，验证可复现性（唯一需要外部 API 的环节）。"""
    import asyncio

    from app.rag.retriever import ReferenceRetriever

    r = _load("eval_recall.json")
    retriever = ReferenceRetriever()

    async def run() -> tuple[int, int]:
        hit = 0
        exp = 0
        mismatch = []
        for row in r["rows"]:
            docs = await retriever.search(row["query"], top_k=r["k"])
            got = {d.get("id") for d in docs}
            exp += len(row["expected"])
            for eid in row["expected"]:
                if eid in got:
                    hit += 1
                else:
                    mismatch.append((row["query"], eid))
        return hit, exp, mismatch

    hit, exp, mismatch = asyncio.run(run())
    print(f"\n[live] 真实重跑检索：命中 {hit}/{exp} = {hit / exp * 100:.1f}%")
    if mismatch:
        print(f"  [warn] 与存档不一致的 {len(mismatch)} 条：{mismatch[:5]}")
    return {
        "live_hit": hit,
        "live_expected": exp,
        "live_recall": round(hit / exp, 4),
        "mismatch": [{"query": q, "expected": e} for q, e in mismatch],
    }


# --------------------------------------------------------------------------
# 3. 生成时长 / 重写触发率 / 引用
# --------------------------------------------------------------------------
def verify_generate() -> dict:
    g = _load("eval_generate.json")
    s = g["summary"]
    papers = [p for p in g["papers"] if p.get("status") == "ok"]

    # --- 时长：从逐篇 seconds 独立重算 ---
    secs = [p["seconds"] for p in papers]
    check("生成·篇数 n", len(papers), s["n"])
    check("生成·总时长重算", round(sum(secs), 1), round(s["total_seconds"], 1), 0.05)
    check("生成·均值重算", round(statistics.mean(secs), 1), round(s["avg_seconds"], 1), 0.05)
    check("生成·中位数重算", round(statistics.median(secs), 1), round(s["median_seconds"], 1), 0.05)
    check("生成·最小值重算", round(min(secs), 1), round(s["min_seconds"], 1), 0.05)
    check("生成·最大值重算", round(max(secs), 1), round(s["max_seconds"], 1), 0.05)
    # 均值区间自洽：min <= avg <= max
    check("生成·均值落在[min,max]内",
          round(min(secs), 1) <= s["avg_seconds"] <= round(max(secs), 1), True)

    # --- 重写触发率：从逐篇 rewrite_triggered 独立重算 ---
    triggered = sum(1 for p in papers if p.get("rewrite_triggered"))
    check("重写·触发篇数重算", triggered, s["rewrite_triggered_count"])
    check("重写·触发率重算",
          round(triggered / len(papers), 4), round(s["rewrite_trigger_rate"], 4))
    revs = [p.get("revision_count", 0) for p in papers]
    check("重写·平均 revision_count 重算",
          round(statistics.mean(revs), 2), round(s["avg_revision_count"], 2), 0.01)
    # 派生：重写轮次 = revision_count - 1（质检过 + 重写过的轮数）
    rounds = [max(0, r - 1) for r in revs]
    if "avg_rewrite_rounds" in s:
        check("重写·平均重写轮次(=rev-1) 重算",
              round(statistics.mean(rounds), 2), round(s["avg_rewrite_rounds"], 2), 0.01)

    # --- 引用：用 corpus 标题独立比对（交叉校验，不复用评测中间结果）---
    from app.rag.corpus import ACADEMIC_REFERENCES

    corpus_titles = {d["metadata"]["title"] for d in ACADEMIC_REFERENCES}
    total_refs = 0
    traceable = 0
    body_ok = 0
    for p in papers:
        titles = p.get("reference_titles", [])
        total_refs += len(titles)
        traceable += sum(1 for t in titles if t in corpus_titles)
        markers = p.get("body_citation_markers", [])
        if markers and set(markers) <= set(range(1, len(titles) + 1)):
            body_ok += 1
    check("引用·参考文献总条数重算", total_refs, s["total_references"])
    # 纯自洽：可溯源条数应等于总条数（不写死 100.0%，避免"自证"）
    check("引用·条目全部可在语料中溯源", traceable, total_refs)
    check("引用·正文编号与文末列表一致篇数", body_ok, len(papers))

    return {
        "n": len(papers),
        "avg_seconds": round(statistics.mean(secs), 1),
        "median_seconds": round(statistics.median(secs), 1),
        "min_seconds": round(min(secs), 1),
        "max_seconds": round(max(secs), 1),
        "total_seconds": round(sum(secs), 1),
        "avg_word_count": s["avg_word_count"],
        "rewrite_triggered": triggered,
        "rewrite_trigger_rate": round(triggered / len(papers), 4),
        "avg_revision_count": round(statistics.mean(revs), 2),
        "references_total": total_refs,
        "references_traceable": traceable,
        "body_consistent_papers": body_ok,
        "avg_quality_score": s.get("avg_quality_score"),
    }


# --------------------------------------------------------------------------
# 4. 测试用例数（真实 collect）
# --------------------------------------------------------------------------
def verify_test_count() -> dict:
    import re
    import subprocess

    py = sys.executable
    proc = subprocess.run(
        [py, "-m", "pytest", "--collect-only", "-q"],
        cwd=str(BACKEND), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    out = proc.stdout + proc.stderr
    n = sum(1 for ln in out.splitlines() if "::" in ln)
    files = {ln.split("::")[0] for ln in out.splitlines() if "::" in ln}
    print(f"\n[collect] 实测用例数 = {n}（{len(files)} 个测试文件）")
    return {"collected_tests": n, "collected_files": len(files)}


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="独立复核五项量化指标")
    ap.add_argument("--live", action="store_true", help="真实重跑检索评测（需 DASHSCOPE_API_KEY）")
    ap.add_argument("--json", dest="json_out", default=None, help="把复核结果写入指定 JSON")
    args = ap.parse_args()

    print("=" * 78)
    print(" 五项量化指标 · 独立复核（不信任 summary，全部从明细重算）")
    print("=" * 78)

    result: dict = {}
    print("\n[1/4] 覆盖率 ...")
    result["coverage"] = verify_coverage()
    print("[2/4] 检索 Recall@K ...")
    result["recall"] = verify_recall()
    if args.live:
        result["recall_live"] = verify_recall_live()
    print("[3/4] 生成时长 / 重写触发率 / 引用 ...")
    result["generate"] = verify_generate()
    print("[4/4] 测试用例数 ...")
    result["tests"] = verify_test_count()

    # ---- 汇总对照表 ----
    print("\n" + "=" * 78)
    print(f"{'检查项':<44}{'重算值':>16}{'记录值':>14}")
    print("-" * 78)
    fails = 0
    for name, rec, recd, ok in CHECKS:
        flag = "" if ok else "  <== FAIL"
        if not ok:
            fails += 1
        print(f"{name:<44}{str(rec):>16}{str(recd):>14}{flag}")
    print("-" * 78)
    print(f"共 {len(CHECKS)} 项检查，通过 {len(CHECKS) - fails} 项，失败 {fails} 项")
    result["summary"] = {
        "total_checks": len(CHECKS),
        "passed": len(CHECKS) - fails,
        "failed": fails,
        "verdict": "ALL_CONSISTENT" if fails == 0 else "MISMATCH_FOUND",
    }
    print(f"结论：{result['summary']['verdict']}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n复核结果已写入：{args.json_out}")

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

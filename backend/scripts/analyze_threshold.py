# analyze_threshold.py —— 用「首检分 vs 重写后分」实测数据标定质量阈值
#
# 背景：质量门禁 QUALITY_PASS_THRESHOLD 的作用是决定「这篇稿件要不要打回重写」。
# 阈值定高了 → 所有稿件必然重写，门禁退化成固定流程（只靠 max_revisions 兜底）；
# 阈值定低了 → 所有稿件都直接放行，门禁等于不存在。
#
# 标定思路（不靠猜）：每篇论文我们都同时有「首检分 first_pass_score」和
# 「重写后分 quality_score」。对任意候选阈值 t，可以推演出该阈值下的真实后果：
#   - first >= t 的稿件：不重写，最终质量 = first
#   - first <  t 的稿件：触发重写，最终质量 = quality_score（实测值，非估算）
# 于是「平均最终质量」和「重写率」都能对每个 t 算出来，选一个兼顾两者的即可。
#
# 用法：
#   python scripts/analyze_threshold.py                 # 默认读 data/eval_generate_v4.json
#   python scripts/analyze_threshold.py path/to/eval.json
import json
import statistics
import sys
from pathlib import Path

DEFAULT_EVAL = Path(__file__).resolve().parents[1] / "data" / "eval_generate_v4_final.json"

# 设计文档给的预期区间：重写触发率应落在 40-60%，否则门禁失去区分意义
TARGET_REWRITE_RATE = 0.6
# 评分的量化步长是 0.01-0.02，小于这个带宽的均值差异属于噪声，不足以作为否决依据
NOISE_BAND = 0.02
GRID = [round(0.30 + i * 0.01, 2) for i in range(0, 46)]   # 0.30 ~ 0.75


def _load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        r for r in data.get("papers", [])
        if r.get("status") == "ok"
        and r.get("first_pass_score") is not None
        and r.get("quality_score") is not None
    ]
    return data, rows


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EVAL
    if not path.exists():
        print(f"找不到评测结果：{path}")
        return 1

    data, rows = _load(path)
    total = len(data.get("papers", []))
    print(f"数据来源：{path.name}")
    print(f"有效样本：{len(rows)}/{total} 篇（需同时有首检分与重写后分，降级篇会被排除）\n")

    if not rows:
        print("没有有效样本，无法标定。请先确认评测跑通、且未发生评审降级。")
        return 1

    # ---------- 1) 逐篇 ----------
    print("逐篇明细")
    print(f"  {'id':>3}  {'首检':>6}  {'重写后':>6}  {'增益':>6}  {'轮次':>4}  标题")
    for r in rows:
        first, final = r["first_pass_score"], r["quality_score"]
        gain = final - first
        flag = "  ← 重写反而降分" if gain < -0.005 else ""
        print(f"  {r['id']:>3}  {first:>6.2f}  {final:>6.2f}  {gain:>+6.2f}  "
              f"{r.get('revision_count', '?'):>4}  {r['title'][:26]}{flag}")

    # ---------- 2) 分布 ----------
    firsts = [r["first_pass_score"] for r in rows]
    finals = [r["quality_score"] for r in rows]
    print("\n首检分分布")
    print(f"  n={len(firsts)}  min={min(firsts):.2f}  max={max(firsts):.2f}  "
          f"median={statistics.median(firsts):.2f}  mean={statistics.mean(firsts):.3f}")
    if len(firsts) > 1:
        print(f"  标准差 {statistics.stdev(firsts):.3f}  极差 {max(firsts) - min(firsts):.2f}")
    print(f"  重写后：min={min(finals):.2f}  max={max(finals):.2f}  mean={statistics.mean(finals):.3f}")

    gains = [r["quality_score"] - r["first_pass_score"] for r in rows]
    worse = sum(1 for g in gains if g < -0.005)
    print(f"  重写平均增益 {statistics.mean(gains):+.3f}；"
          f"其中 {worse}/{len(gains)} 篇重写后反而降分"
          + ("（best_score 保护会保留更优版本，故最终分不受影响）" if worse else ""))

    # ---------- 3) 阈值推演 ----------
    print("\n阈值推演（按实测数据模拟：首检 ≥ t 直接放行，< t 走重写并取实测重写后分）")
    print(f"  {'阈值':>5}  {'重写率':>7}  {'预估平均质量':>12}  说明")
    best_t, best_mean = None, -1.0
    feasible = []
    for t in GRID:
        rewritten = [r for r in rows if r["first_pass_score"] < t]
        rate = len(rewritten) / len(rows)
        mean_q = statistics.mean(
            [r["quality_score"] if r["first_pass_score"] < t else r["first_pass_score"] for r in rows]
        )
        if rate <= TARGET_REWRITE_RATE:
            feasible.append((t, rate, mean_q))
        note = ""
        if rate == 0:
            note = "门禁失效：全部直接放行"
        elif rate == 1:
            note = "门禁失效：全部重写"
        print(f"  {t:>5.2f}  {rate * 100:>6.0f}%  {mean_q:>12.3f}  {note}")

    # 在「重写率不超过设计目标」的可行域内，取预估质量最高的阈值
    if feasible:
        best_t, best_rate, best_mean = max(feasible, key=lambda x: (x[2], -x[1]))

    # ---------- 4) 结论 ----------
    print("\n" + "=" * 62)
    print("结论")
    print("=" * 62)
    current = _current_threshold()
    cur_rewrite = sum(1 for r in rows if r["first_pass_score"] < current) / len(rows)
    cur_mean = statistics.mean(
        [r["quality_score"] if r["first_pass_score"] < current else r["first_pass_score"] for r in rows]
    )
    print(f"  当前阈值        : {current:.2f}（重写率 {cur_rewrite * 100:.0f}%，"
          f"预估平均质量 {cur_mean:.3f}）")

    if best_t is None:
        print("  无法给出推荐：重写率在任何阈值下都超出目标区间，样本可能整体偏低。")
        print("  建议：先确认这批评测样本是否代表真实用户稿件；若确实偏低，")
        print("        应优先提升生成质量（大纲/写作环节），而不是继续压低阈值。")
        return 0

    delta = best_mean - cur_mean
    print(f"  最优候选        : {best_t:.2f}（重写率 {best_rate * 100:.0f}%，"
          f"预估平均质量 {best_mean:.3f}，较当前 {delta:+.3f}）")

    # 质量下降的候选一律不推荐：省下的重写成本换不回成品质量，得不偿失
    if delta < -NOISE_BAND:
        print("  → 不建议调整。该阈值虽降低重写率，但成品平均质量下降 "
              f"{abs(delta):.3f}（超过 {NOISE_BAND:.2f} 噪声带），属于用质量换成本。")
        print("  → 现有数据反而说明当前重写是有效的：重写平均增益 "
              f"{statistics.mean([r['quality_score'] - r['first_pass_score'] for r in rows]):+.3f}。")
        print("  → 若目标是降低重写率，应提升首检通过率（生成质量），而非放宽阈值。")
    elif best_rate < cur_rewrite - 0.2:
        # 质量差异在噪声带内，而重写率显著下降 → 门禁从「必然重写」变成「真能放行」
        print(f"  → 建议调整到 {best_t:.2f}：平均质量变化 {delta:+.3f} 在噪声带内，"
              f"重写率 {cur_rewrite * 100:.0f}% → {best_rate * 100:.0f}%。")
        print("  → 关键收益不是质量，而是门禁恢复区分能力："
              f"当前 {cur_rewrite * 100:.0f}% 重写意味着每一次质检都必然打回，阈值形同虚设。")
    elif abs(best_t - current) < 0.005:
        print("  → 当前阈值已是最优，无需调整。")
    elif delta > 0.002:
        print(f"  → 建议调整到 {best_t:.2f}：预估平均质量提升 {delta:+.3f}，"
              f"重写率 {cur_rewrite * 100:.0f}% → {best_rate * 100:.0f}%。")
    else:
        print("  → 调整后平均质量基本持平，主要收益是重写成本下降，可按需调整。")

    if len(rows) < 6:
        print(f"\n  ⚠ 样本仅 {len(rows)} 篇，统计意义有限：建议 ≥6 篇（最好 10+）后再定稿。")
    if max(firsts) < best_t:
        print("  ⚠ 所有首检分都低于推荐阈值，门禁仍会 100% 触发，请复核。")
    return 0


def _current_threshold() -> float:
    """读当前生效阈值（配置项，可被 .env 覆盖）"""
    try:
        from app.config import settings
        return float(settings.QUALITY_PASS_THRESHOLD)
    except Exception:
        return 0.44


if __name__ == "__main__":
    raise SystemExit(main())

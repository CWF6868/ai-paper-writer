# scripts/report_llm_cost.py —— 跑一篇真实论文，然后从 /api/metrics/llm 出成本账单
#
# 为什么走 HTTP 接口而不是直接查表：
#   成本归因链路有三个环节（上下文打标 → 回调计量 → 异步落库），
#   任何一环断了，表里都可能"看着有数但归不到人"。直接查表测不出这一点，
#   而走接口会同时暴露 user_id 归因、聚合口径和落库自检。
#
# 用法：
#   ./venv/Scripts/python.exe scripts/report_llm_cost.py
#   ./venv/Scripts/python.exe scripts/report_llm_cost.py --skip-generate   # 只读已有数据
import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import Base, engine  # noqa: E402
import app.models.user, app.models.paper, app.models.chapter, app.models.reference  # noqa: E402,F401
import app.models.llm_call_log  # noqa: E402,F401
from app.core.llm_log_sink import usage_log_sink  # noqa: E402
from app.main import _ensure_llm_log_columns, _ensure_quality_columns, app  # noqa: E402

PASSWORD = "test123456"
TOPIC = {
    "title": "基于检索增强生成的大语言模型幻觉缓解方法研究",
    "topic": "检索增强生成（RAG）与幻觉缓解",
    "keywords": "RAG,幻觉缓解,大语言模型",
}


async def _login(c: AsyncClient, username: str) -> dict:
    r = await c.post("/api/auth/register", json={
        "username": username, "email": f"{username}@example.com", "password": PASSWORD,
    })
    if r.status_code not in (201, 400):      # 400 = 已存在，直接登录
        raise SystemExit(f"注册失败：{r.status_code} {r.text}")
    r = await c.post("/api/auth/login", data={"username": username, "password": PASSWORD})
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _money(x) -> str:
    return f"{x:.6f}"


def _print_report(d: dict) -> None:
    t, s = d["totals"], d["scope"]
    who = "全平台" if s["global_view"] else f"用户 {s['user_id']}"
    end = s["end"][:16].replace("T", " ")
    print("\n" + "=" * 78)
    print(f"范围：{who} | 最近 {s['days']} 天 | {s['start'][:10]} → {end}（北京时间）")
    print("=" * 78)
    print(f"调用 {t['calls']} 次（失败 {t['errors']}）| 论文 {t['papers']} 篇 | "
          f"无论文归属 {t['calls_without_paper']} 次")
    print(f"Token：输入 {t['prompt_tokens']:,} / 输出 {t['completion_tokens']:,} / "
          f"缓存命中 {t['cache_hit_tokens']:,}（命中率 {t['cache_hit_rate']:.1%}）")
    print(f"总成本 {_money(t['cost'])} 元"
          + (f" | 单篇 {_money(t['cost_per_paper'])} 元" if t["cost_per_paper"] else ""))
    print(f"延迟　平均 {t['avg_latency_ms']/1000:.1f}s / P95 {t['p95_latency_ms']/1000:.1f}s")
    print(f"省钱项：缓存命中省 {_money(t['cache_savings'])} 元 | "
          f"错峰可再省 {_money(t['cost_if_all_peak'] - t['cost'])} 元"
          f"（全峰时成本 {_money(t['cost_if_all_peak'])} 元）")
    print(f"峰谷　峰时 {_money(t['peak_cost'])} / 空闲 {_money(t['off_peak_cost'])}")

    for title, rows, key_name in (
        ("按节点", d["by_node"], "节点"),
        ("按模型", d["by_model"], "模型"),
        ("按天", d["by_day"], "日期"),
    ):
        if not rows:
            continue
        print(f"\n{title}")
        print(f"  {key_name:<20}{'次':>4}{'输入':>9}{'输出':>8}{'命中':>8}{'成本(元)':>11}{'占比':>8}{'平均延迟':>10}")
        for b in rows:
            print(f"  {b['key']:<20}{b['calls']:>4}{b['prompt_tokens']:>9,}"
                  f"{b['completion_tokens']:>8,}{b['cache_hit_tokens']:>8,}"
                  f"{b['cost']:>11.6f}{b['cost_share']:>7.1%}{b['avg_latency_ms']/1000:>9.1f}s")

    ing = d["ingest"]
    print(f"\n落库自检：收到 {ing['received']} / 写入 {ing['written']} / 待写 {ing['pending']} / "
          f"丢弃 {ing['dropped']} → {'账平' if ing['balanced'] else '⚠ 账不平，成本表不可信'}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default="cost_demo", help="用于跑真实生成的演示账号")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--skip-generate", action="store_true", help="不跑生成，只读已有数据")
    ap.add_argument("--out", default="data/llm_cost_report.json")
    args = ap.parse_args()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_quality_columns(conn)
        await _ensure_llm_log_columns(conn)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=1200) as c:
        h = await _login(c, args.username)

        if not args.skip_generate:
            payload = {**TOPIC, "title": f"{TOPIC['title']}（{uuid.uuid4().hex[:4]}）"}
            r = await c.post("/api/papers/", json=payload, headers=h)
            r.raise_for_status()
            paper_id = r.json()["id"]
            print(f"论文 #{paper_id} 开始生成……（真实模型调用，约 1-2 分钟）")

            t0 = time.perf_counter()
            r = await c.post(f"/api/papers/{paper_id}/generate", headers=h)
            elapsed = time.perf_counter() - t0
            r.raise_for_status()
            body = r.json()
            print(f"完成：{elapsed:.1f}s | 章节 {body['sections']} | 文献 {body['references']} | "
                  f"质量分 {body['quality_score']}")

        await usage_log_sink.stop()     # 把队列里最后一批写进表，再读指标

        r = await c.get(f"/api/metrics/llm?days={args.days}", headers=h)
        r.raise_for_status()
        data = r.json()

    _print_report(data)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存：{out}")


if __name__ == "__main__":
    asyncio.run(main())

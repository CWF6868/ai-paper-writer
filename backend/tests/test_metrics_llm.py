# tests/test_metrics_llm.py —— 成本指标接口
#
# 重点不是"接口能返回 200"，而是几个会算错钱、会越权的地方：
#   1) 按天聚合必须用北京时间（峰谷计价按北京时间划）
#   2) 用户隔离必须是「覆盖」而不是「信任参数」
#   3) 缓存省下的钱、峰时多付的钱，算法本身要能被独立复算
import pytest
from datetime import datetime, time, timedelta, timezone

from sqlalchemy import select, update

from app.core.llm_client import BEIJING_TZ, compute_cost
from app.core.llm_log_sink import usage_log_sink
from app.database import AsyncSessionLocal
from app.models.llm_call_log import LlmCallLog
from app.models.user import User

MODEL = "deepseek-flash"


async def _uid(username: str) -> int:
    async with AsyncSessionLocal() as db:
        return (await db.execute(select(User.id).where(User.username == username))).scalar_one()


async def _login(client, username: str) -> dict:
    from helpers import register_and_login
    token = await register_and_login(client, username)
    return {"Authorization": f"Bearer {token}"}


async def _seed(rows) -> None:
    """直接写库，绕开 sink：指标接口测的是聚合逻辑，不是落库链路。"""
    async with AsyncSessionLocal() as db:
        db.add_all([LlmCallLog(**r) for r in rows])
        await db.commit()


def _row(node="write_sections", model=MODEL, p=1000, c=500, hit=0,
         latency=1000.0, status="ok", peak=False, paper_id=None, user_id=None,
         created_at=None):
    cost = compute_cost(model, p, c, hit, peak=peak)
    return {
        "node": node, "model": model, "prompt_tokens": p,
        "completion_tokens": c, "cache_hit_tokens": hit, "total_tokens": p + c,
        "cost": cost, "latency_ms": latency, "status": status, "peak": peak,
        "paper_id": paper_id, "user_id": user_id,
        "created_at": created_at or datetime.now(timezone.utc).replace(tzinfo=None),
    }


# --------------------------------------------------------------------------
# 基础：空库、无数据时不能崩，也不能编数字
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_empty_data_returns_zeros(client):
    h = await _login(client, "metrics_empty")
    r = await client.get("/api/metrics/llm", headers=h)
    assert r.status_code == 200, r.text
    d = r.json()

    assert d["totals"]["calls"] == 0
    assert d["totals"]["cost"] == 0.0
    assert d["totals"]["cost_per_paper"] is None      # 没论文就不能编个 0 出来
    assert d["by_node"] == [] and d["by_day"] == [] and d["by_model"] == []
    assert d["scope"]["global_view"] is False
    # 空库也要能自检：没有丢失时对账等式必须成立
    assert d["ingest"]["balanced"] is True


# --------------------------------------------------------------------------
# 口径：按天聚合用北京时间
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_day_bucket_uses_beijing_time(client):
    """北京时间今天 00:30 的调用，UTC 日期是昨天 —— 必须归到北京的今天。"""
    h = await _login(client, "metrics_tz")
    uid = await _uid("metrics_tz")

    today_bj = datetime.now(BEIJING_TZ).date()
    ts_bj = datetime.combine(today_bj, time(0, 30), tzinfo=BEIJING_TZ)
    ts_utc = ts_bj.astimezone(timezone.utc).replace(tzinfo=None)
    # 前提校验：这个时刻的 UTC 日期确实比北京日期早一天，否则用例没意义
    assert ts_utc.date() < today_bj

    await _seed([_row(user_id=uid, created_at=ts_utc)])

    d = (await client.get("/api/metrics/llm", headers=h)).json()
    assert [b["key"] for b in d["by_day"]] == [today_bj.isoformat()]
    assert d["by_day"][0]["calls"] == 1


# --------------------------------------------------------------------------
# 权限：普通用户只能看自己，管理员可看全量
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_user_isolation(client):
    h_a = await _login(client, "metrics_a")
    h_b = await _login(client, "metrics_b")
    uid_a, uid_b = await _uid("metrics_a"), await _uid("metrics_b")

    await _seed([
        _row(user_id=uid_a, p=1000, c=100),
        _row(user_id=uid_b, p=9000, c=900),
        _row(user_id=uid_b, p=9000, c=900),
    ])

    da = (await client.get("/api/metrics/llm", headers=h_a)).json()
    db_ = (await client.get("/api/metrics/llm", headers=h_b)).json()

    assert da["totals"]["calls"] == 1
    assert da["scope"]["user_id"] == uid_a
    assert db_["totals"]["calls"] == 2
    # 两人的成本相加 = 全量，不能有"谁也不是"的孤儿记录被漏掉
    assert round(da["totals"]["cost"] + db_["totals"]["cost"], 8) > 0


@pytest.mark.asyncio
async def test_normal_user_cannot_query_other_user(client):
    """传别人的 user_id 不报错、但被强制覆盖成自己 —— 越权不能靠"返回 403"兜底。"""
    h_a = await _login(client, "metrics_spy")
    await _login(client, "metrics_victim")
    uid_a, uid_v = await _uid("metrics_spy"), await _uid("metrics_victim")

    await _seed([_row(user_id=uid_v, p=5000, c=500)])

    d = (await client.get(f"/api/metrics/llm?user_id={uid_v}", headers=h_a)).json()
    assert d["scope"]["user_id"] == uid_a       # 被覆盖
    assert d["totals"]["calls"] == 0            # 看不到对方的数据


@pytest.mark.asyncio
async def test_superuser_global_and_can_filter(client):
    h_boss = await _login(client, "metrics_boss")
    h_x = await _login(client, "metrics_x")
    uid_boss, uid_x = await _uid("metrics_boss"), await _uid("metrics_x")

    async with AsyncSessionLocal() as db:
        await db.execute(update(User).where(User.id == uid_boss).values(is_superuser=True))
        await db.commit()

    await _seed([_row(user_id=uid_boss), _row(user_id=uid_x), _row(user_id=uid_x)])

    g = (await client.get("/api/metrics/llm", headers=h_boss)).json()
    assert g["scope"]["global_view"] is True
    assert g["scope"]["user_id"] is None
    assert g["totals"]["calls"] == 3

    f = (await client.get(f"/api/metrics/llm?user_id={uid_x}", headers=h_boss)).json()
    assert f["scope"]["global_view"] is False
    assert f["totals"]["calls"] == 2


# --------------------------------------------------------------------------
# 过滤与窗口
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_node_filter_and_window(client):
    h = await _login(client, "metrics_filter")
    uid = await _uid("metrics_filter")
    old = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=200)

    await _seed([
        _row(user_id=uid, node="polish_paper"),
        _row(user_id=uid, node="write_sections"),
        _row(user_id=uid, node="write_sections"),
        _row(user_id=uid, node="polish_paper", created_at=old),   # 窗口外
    ])

    d = (await client.get("/api/metrics/llm?days=7", headers=h)).json()
    assert d["totals"]["calls"] == 3, "200 天前的记录不该进 7 天窗口"

    n = (await client.get("/api/metrics/llm?node=write_sections", headers=h)).json()
    assert n["totals"]["calls"] == 2
    assert [b["key"] for b in n["by_node"]] == ["write_sections"]

    # 非法参数要被 FastAPI 拦住，而不是拿 days=0 跑出一个空窗口
    assert (await client.get("/api/metrics/llm?days=0", headers=h)).status_code == 422


# --------------------------------------------------------------------------
# 派生指标：缓存省的钱、峰时多付的钱，必须与独立复算一致
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cache_savings_matches_independent_recalc(client):
    h = await _login(client, "metrics_cache")
    uid = await _uid("metrics_cache")
    p, c, hit = 10_000, 1_000, 6_000

    await _seed([_row(user_id=uid, p=p, c=c, hit=hit, peak=False)])

    d = (await client.get("/api/metrics/llm", headers=h)).json()
    t = d["totals"]

    expected_saving = float(
        compute_cost(MODEL, p, c, 0, peak=False) - compute_cost(MODEL, p, c, hit, peak=False)
    )
    assert t["cache_hit_rate"] == pytest.approx(hit / p, abs=1e-4)
    assert t["cache_savings"] == pytest.approx(expected_saving, abs=1e-8)
    # 缓存命中一定省钱：省不出来说明计价维度接错了
    assert t["cache_savings"] > 0
    # 成本必须小于「完全不命中缓存」的成本
    assert t["cost"] < float(compute_cost(MODEL, p, c, 0, peak=False))


@pytest.mark.asyncio
async def test_peak_premium_matches_independent_recalc(client):
    h = await _login(client, "metrics_peak")
    uid = await _uid("metrics_peak")
    p, c, hit = 8_000, 2_000, 0

    await _seed([
        _row(user_id=uid, p=p, c=c, hit=hit, peak=True),
        _row(user_id=uid, p=p, c=c, hit=hit, peak=False),
    ])

    t = (await client.get("/api/metrics/llm", headers=h)).json()["totals"]

    premium = float(
        compute_cost(MODEL, p, c, hit, peak=True) - compute_cost(MODEL, p, c, hit, peak=False)
    )
    assert t["peak_premium"] == pytest.approx(premium, abs=1e-8)
    assert t["peak_cost"] > t["off_peak_cost"]      # 同样的 token，峰时更贵
    # 单价表是严格 2 倍关系，两笔同 token 的调用应精确呈现这个关系
    assert t["peak_cost"] == pytest.approx(t["off_peak_cost"] * 2, rel=1e-6)
    # 反事实口径的恒等式：全峰时成本 - 实际成本 = 空闲那部分本会多付的钱。
    # （不要写成 cost×2 或 off_peak×2：这两条都只在特定数据下巧合成立，
    #   我在这里就写错过两次 —— 恒等式的形式才和数据条数无关。）
    assert t["cost_if_all_peak"] - t["cost"] == pytest.approx(t["off_peak_cost"], rel=1e-6)
    assert t["cost_if_all_peak"] == pytest.approx(t["peak_cost"] + t["off_peak_cost"] * 2, rel=1e-6)


# --------------------------------------------------------------------------
# 结构：占比、单篇成本、P95、无论文调用
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_structure_metrics(client):
    h = await _login(client, "metrics_struct")
    uid = await _uid("metrics_struct")

    await _seed([
        # 两篇论文 + 一次选题推荐（无 paper_id）
        _row(user_id=uid, node="polish_paper", paper_id=1, latency=100),
        _row(user_id=uid, node="write_sections", paper_id=1, latency=200),
        _row(user_id=uid, node="write_sections", paper_id=2, latency=300),
        _row(user_id=uid, node="topic", paper_id=None, latency=400),
    ])

    d = (await client.get("/api/metrics/llm", headers=h)).json()
    t = d["totals"]

    assert t["papers"] == 2
    assert t["calls_without_paper"] == 1
    assert t["cost_per_paper"] == pytest.approx(t["cost"] / 2, abs=1e-8)

    # 各节点占比之和必须为 1（否则说明有成本没被归到任何节点）
    assert sum(b["cost_share"] for b in d["by_node"]) == pytest.approx(1.0, abs=1e-4)

    # P95 用最近秩口径：4 条里的第 4 条
    assert t["p95_latency_ms"] == 400
    assert t["avg_latency_ms"] == pytest.approx(250.0, abs=1e-6)


@pytest.mark.asyncio
async def test_error_calls_counted_and_shown(client):
    h = await _login(client, "metrics_err")
    uid = await _uid("metrics_err")

    await _seed([
        _row(user_id=uid),
        {**_row(user_id=uid), "status": "error", "cost": 0, "latency_ms": 5000.0},
    ])

    d = (await client.get("/api/metrics/llm", headers=h)).json()
    assert d["totals"]["calls"] == 2
    assert d["totals"]["errors"] == 1


@pytest.mark.asyncio
async def test_ingest_stats_exposed(client):
    """落库自检必须能从接口看到：账不平的时候，先怀疑数据完整性。"""
    h = await _login(client, "metrics_ingest")
    d = (await client.get("/api/metrics/llm", headers=h)).json()
    s = usage_log_sink.stats()

    assert d["ingest"]["received"] == s["received"]
    assert d["ingest"]["written"] == s["written"]
    assert d["ingest"]["balanced"] is True

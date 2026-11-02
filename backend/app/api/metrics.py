# api/metrics.py —— LLM 成本指标（P0-3 成本故事的对外出口）
#
# 三个聚合维度是分开设计的，因为它们是三个不同的问题：
#   by_node  —— 钱花在哪一步（决定该优化谁）
#   by_day   —— 哪天花的（决定预算和趋势）
#   by_model —— 是不是有模型跑错/单价缺失（决定账是否可信）
#
# 另外两个派生指标是这个接口真正的价值，而不是"把 SUM 包一层"：
#   cache_savings —— 缓存命中的钱，不计量就会被当成"本来就这么便宜"
#   peak_premium  —— 峰时溢价，即"同样的事放到晚上跑能省多少"，是可执行的动作
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.core.llm_client import BEIJING_TZ, compute_cost
from app.core.llm_log_sink import usage_log_sink
from app.database import get_db
from app.models.llm_call_log import LlmCallLog
from app.models.user import User
from app.schemas.metrics import (
    IngestStats, LlmMetricsResponse, MetricBucket, MetricScope, MetricTotals,
)

router = APIRouter()

# 按天聚合必须用北京时间：峰谷计价本身就是按北京时间划的，
# 用 UTC 分天会把 08:00-09:00（北京）这段峰时算到前一天，成本结构直接看歪。
_DAY_EXPR = func.date(LlmCallLog.created_at, "+8 hours")

_ERR = case((LlmCallLog.status != "ok", 1), else_=0)
_NO_PAPER = case((LlmCallLog.paper_id.is_(None), 1), else_=0)


def _sum(col):
    return func.coalesce(func.sum(col), 0)


def _window(days: int):
    """把「最近 N 天」换算成 UTC 查询区间 + 北京时间展示区间。

    返回的 start_utc 是 naive 的：SQLite 的 DateTime 列存的是 UTC 墙钟时间
    （写入时 tzinfo 被丢弃），拿带时区的值去比会因字符串格式不一致而漏行。
    """
    now_bj = datetime.now(BEIJING_TZ)
    start_bj = (now_bj - timedelta(days=days - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_utc = start_bj.astimezone(timezone.utc).replace(tzinfo=None)
    return start_bj, now_bj, start_utc


def _bucket(row, total_cost: float) -> MetricBucket:
    cost = float(row.cost or 0)
    return MetricBucket(
        key=str(row.key),
        calls=int(row.calls or 0),
        errors=int(row.errors or 0),
        prompt_tokens=int(row.prompt_tokens or 0),
        completion_tokens=int(row.completion_tokens or 0),
        cache_hit_tokens=int(row.cache_hit_tokens or 0),
        total_tokens=int(row.total_tokens or 0),
        cost=round(cost, 8),
        avg_latency_ms=round(float(row.avg_latency_ms or 0), 1),
        cost_share=round(cost / total_cost, 4) if total_cost > 0 else 0.0,
    )


@router.get("/llm", response_model=LlmMetricsResponse, summary="LLM 调用成本与延迟聚合")
async def llm_metrics(
    days: int = Query(7, ge=1, le=90, description="统计最近 N 天（含今天，北京时间自然日）"),
    node: Optional[str] = Query(None, description="只看某个节点的调用"),
    user_id: Optional[int] = Query(None, description="管理员可指定用户；普通用户传了也无效"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """按节点 / 按天 / 按模型聚合 token、成本、延迟。

    权限：普通用户只能看自己的（强制覆盖 user_id）；管理员可看全量或指定用户。
    """
    is_admin = bool(getattr(current_user, "is_superuser", False))
    # 普通用户无论传什么，都只能看自己的 —— 这里做的是"覆盖"而不是"校验后拒绝"，
    # 因为拒绝会把内部字段暴露成一个可探测的 400，而覆盖不会泄露任何信息。
    effective_user = user_id if (is_admin and user_id is not None) else (
        None if is_admin else current_user.id
    )

    start_bj, end_bj, start_utc = _window(days)

    conditions = [LlmCallLog.created_at >= start_utc]
    if effective_user is not None:
        conditions.append(LlmCallLog.user_id == effective_user)
    if node:
        conditions.append(LlmCallLog.node == node)

    # ---------- 1) 总量 ----------
    totals_row = (await db.execute(
        select(
            func.count().label("calls"),
            _sum(_ERR).label("errors"),
            _sum(LlmCallLog.prompt_tokens).label("prompt_tokens"),
            _sum(LlmCallLog.completion_tokens).label("completion_tokens"),
            _sum(LlmCallLog.cache_hit_tokens).label("cache_hit_tokens"),
            _sum(LlmCallLog.total_tokens).label("total_tokens"),
            _sum(LlmCallLog.cost).label("cost"),
            func.coalesce(func.avg(LlmCallLog.latency_ms), 0).label("avg_latency_ms"),
            func.count(distinct(LlmCallLog.paper_id)).label("papers"),
            _sum(_NO_PAPER).label("calls_without_paper"),
        ).where(*conditions)
    )).one()

    calls = int(totals_row.calls or 0)
    total_cost = float(totals_row.cost or 0)
    prompt_tokens = int(totals_row.prompt_tokens or 0)
    cache_hit = int(totals_row.cache_hit_tokens or 0)

    # ---------- 2) P95 延迟 ----------
    # SQLite 没有 percentile 函数，用排序 + offset 取第 95 分位（口径写清楚，
    # 别让人误以为是插值百分位）。
    p95 = 0.0
    if calls:
        offset = max(0, int(round(calls * 0.95)) - 1)
        p95 = float((await db.execute(
            select(LlmCallLog.latency_ms).where(*conditions)
            .order_by(LlmCallLog.latency_ms).offset(offset).limit(1)
        )).scalar_one_or_none() or 0.0)

    # ---------- 3) 按节点 / 按天 / 按模型 ----------
    common = (
        func.count().label("calls"),
        _sum(_ERR).label("errors"),
        _sum(LlmCallLog.prompt_tokens).label("prompt_tokens"),
        _sum(LlmCallLog.completion_tokens).label("completion_tokens"),
        _sum(LlmCallLog.cache_hit_tokens).label("cache_hit_tokens"),
        _sum(LlmCallLog.total_tokens).label("total_tokens"),
        _sum(LlmCallLog.cost).label("cost"),
        func.coalesce(func.avg(LlmCallLog.latency_ms), 0).label("avg_latency_ms"),
    )

    node_rows = (await db.execute(
        select(LlmCallLog.node.label("key"), *common)
        .where(*conditions)
        .group_by(LlmCallLog.node)
        .order_by(_sum(LlmCallLog.cost).desc())
    )).all()

    day_rows = (await db.execute(
        select(_DAY_EXPR.label("key"), *common)
        .where(*conditions)
        .group_by(_DAY_EXPR)
        .order_by(_DAY_EXPR)
    )).all()

    model_rows = (await db.execute(
        select(LlmCallLog.model.label("key"), *common)
        .where(*conditions)
        .group_by(LlmCallLog.model)
        .order_by(_sum(LlmCallLog.cost).desc())
    )).all()

    # ---------- 4) 派生指标：缓存省了多少、峰时贵了多少 ----------
    # 按 (model, peak) 分组再套单价，而不是逐行算：
    # 单价对 token 是线性的，所以「先按同价组求和、再乘单价」与逐行累加等价，
    # 但只产生个位数的分组，不用把上万行拉进 Python。
    price_rows = (await db.execute(
        select(
            LlmCallLog.model,
            LlmCallLog.peak,
            _sum(LlmCallLog.prompt_tokens).label("prompt_tokens"),
            _sum(LlmCallLog.completion_tokens).label("completion_tokens"),
            _sum(LlmCallLog.cache_hit_tokens).label("cache_hit_tokens"),
            _sum(LlmCallLog.cost).label("cost"),
        ).where(*conditions).group_by(LlmCallLog.model, LlmCallLog.peak)
    )).all()

    cache_savings = 0.0
    peak_premium = 0.0
    peak_cost = 0.0
    off_peak_cost = 0.0
    cost_if_all_peak = 0.0
    for r in price_rows:
        p, c, hit = int(r.prompt_tokens or 0), int(r.completion_tokens or 0), int(r.cache_hit_tokens or 0)
        is_peak = bool(r.peak)
        actual = compute_cost(r.model, p, c, hit, peak=is_peak)
        no_cache = compute_cost(r.model, p, c, 0, peak=is_peak)
        cache_savings += float(no_cache - actual)
        # 反事实：这批 token 若全落在峰时会花多少。与真实成本之差就是错峰的价值。
        # 比 peak_premium 有用，因为后者只在"确实峰时跑过"时才非零。
        cost_if_all_peak += float(compute_cost(r.model, p, c, hit, peak=True))
        if is_peak:
            peak_cost += float(r.cost or 0)
            peak_premium += float(actual - compute_cost(r.model, p, c, hit, peak=False))
        else:
            off_peak_cost += float(r.cost or 0)

    totals = MetricTotals(
        calls=calls,
        errors=int(totals_row.errors or 0),
        prompt_tokens=prompt_tokens,
        completion_tokens=int(totals_row.completion_tokens or 0),
        cache_hit_tokens=cache_hit,
        total_tokens=int(totals_row.total_tokens or 0),
        cost=round(total_cost, 8),
        avg_latency_ms=round(float(totals_row.avg_latency_ms or 0), 1),
        p95_latency_ms=round(p95, 1),
        papers=int(totals_row.papers or 0),
        calls_without_paper=int(totals_row.calls_without_paper or 0),
        cost_per_paper=round(total_cost / int(totals_row.papers), 8) if totals_row.papers else None,
        cache_hit_rate=round(cache_hit / prompt_tokens, 4) if prompt_tokens else 0.0,
        cache_savings=round(cache_savings, 8),
        peak_cost=round(peak_cost, 8),
        off_peak_cost=round(off_peak_cost, 8),
        peak_premium=round(peak_premium, 8),
        cost_if_all_peak=round(cost_if_all_peak, 8),
    )

    # ---------- 5) 落库自检 ----------
    s = usage_log_sink.stats()
    dropped = s["dropped_full"] + s["dropped_no_loop"] + s["dropped_flush"]
    ingest = IngestStats(
        received=s["received"],
        written=s["written"],
        pending=s["pending"],
        dropped=dropped,
        flush_failures=s["flush_failures"],
        balanced=s["balanced"],
        note="进程内计数，重启归零；balanced=False 表示成本表存在丢失，数字不可信",
    )

    return LlmMetricsResponse(
        scope=MetricScope(
            user_id=effective_user,
            global_view=is_admin and effective_user is None,
            days=days,
            start=start_bj.isoformat(),
            end=end_bj.isoformat(),
            node=node,
        ),
        totals=totals,
        by_node=[_bucket(r, total_cost) for r in node_rows],
        by_day=[_bucket(r, total_cost) for r in day_rows],
        by_model=[_bucket(r, total_cost) for r in model_rows],
        ingest=ingest,
    )

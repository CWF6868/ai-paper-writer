# schemas/metrics.py —— 成本指标接口的响应模型
#
# 为什么字段要这么细：只回一个"总花费"没法指导行动。
# 要回答的是「钱花在哪个节点、哪天、能不能错峰、缓存省了多少、
# 有没有记录没落库导致账不平」，所以每个口径都单列出来。
from typing import List, Optional

from pydantic import BaseModel, Field


class MetricScope(BaseModel):
    """这次查询的可见范围与时间窗口。返回它，是为了让调用方知道
    「我看的是全量还是自己的」——数字对不上时先怀疑范围而不是怀疑数据。"""
    user_id: Optional[int] = Field(None, description="实际生效的用户过滤；None = 全量")
    global_view: bool = Field(..., description="True = 全平台数据（仅管理员）")
    days: int
    start: str = Field(..., description="窗口起点（北京时间 ISO）")
    end: str = Field(..., description="窗口终点（北京时间 ISO，即当前时刻）")
    timezone: str = "Asia/Shanghai (UTC+8)"
    node: Optional[str] = Field(None, description="按节点过滤，None = 全部节点")


class MetricBucket(BaseModel):
    """一个分组（节点 / 日期 / 模型）的汇总。"""
    key: str
    calls: int
    errors: int
    prompt_tokens: int
    completion_tokens: int
    cache_hit_tokens: int
    total_tokens: int
    cost: float = Field(..., description="元")
    avg_latency_ms: float
    cost_share: float = Field(..., description="占总成本比例 0-1，用于直接看结构")


class MetricTotals(BaseModel):
    calls: int
    errors: int
    prompt_tokens: int
    completion_tokens: int
    cache_hit_tokens: int
    total_tokens: int
    cost: float
    avg_latency_ms: float
    p95_latency_ms: float

    papers: int = Field(..., description="涉及的论文数（按 paper_id 去重）")
    calls_without_paper: int = Field(..., description="无法归属到论文的调用数，如选题推荐")
    cost_per_paper: Optional[float] = Field(None, description="单篇平均成本；无论文数据时为 None")

    cache_hit_rate: float = Field(..., description="缓存命中 token / 输入 token")
    cache_savings: float = Field(..., description="缓存命中相对「全部按未命中计价」省下的钱")
    peak_cost: float = Field(..., description="峰时时段产生的成本")
    off_peak_cost: float = Field(..., description="空闲时段产生的成本")
    peak_premium: float = Field(..., description="峰时溢价：已经因为落在峰时多付的钱")
    cost_if_all_peak: float = Field(
        ..., description="同样 token 全部落在峰时的成本；与 cost 之差 = 错峰省下的钱"
    )


class IngestStats(BaseModel):
    """落库自检。balanced=False 时，成本表的数字不可信 —— 有记录在链路上丢了。"""
    received: int
    written: int
    pending: int
    dropped: int
    flush_failures: int
    balanced: bool
    note: str = Field(..., description="口径说明：进程内计数，重启归零")


class LlmMetricsResponse(BaseModel):
    scope: MetricScope
    totals: MetricTotals
    by_node: List[MetricBucket]
    by_day: List[MetricBucket]
    by_model: List[MetricBucket]
    ingest: IngestStats

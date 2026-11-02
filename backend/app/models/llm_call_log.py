# models/llm_call_log.py —— 模型调用流水表（P0-3 成本归因）
#
# 设计要点：
#   1) 不加外键到 papers：论文被删除后，成本流水仍要留存（否则历史成本凭空消失，
#      对账时对不上账单）。paper_id 只做普通索引。
#   2) cost 用 Numeric(18,8) 而不是 Float：单次调用常在 1e-5 元量级，
#      Float 累加上万条会漂移到看不清；8 位小数刚好覆盖最便宜的一次调用。
#   3) created_at 存 UTC（与项目其余表一致），按天聚合时在第 3 步换算北京时间 ——
#      峰谷计价本来就是按北京时间划的，聚合口径不能混。
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Index, Integer, Numeric, String,
)

from app.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class LlmCallLog(Base):
    """一次模型调用一行。字段与 core/llm_client.UsageRecord 一一对应。"""

    __tablename__ = "llm_call_log"

    id = Column(Integer, primary_key=True, index=True)

    paper_id = Column(Integer, index=True)          # 归属论文；选题推荐等无论文场景为空
    # 归属用户。按用户看成本只能靠这一列：paper_id 无外键（论文删除后流水要留存），
    # 反查 papers 会漏掉「已删除论文」和「无 paper_id 的选题调用」两类成本。
    user_id = Column(Integer, index=True)
    node = Column(String(50), nullable=False, index=True)   # 工作流节点名，成本归因的主维度
    model = Column(String(100), nullable=False)             # 响应里的实际模型名（非配置值）

    prompt_tokens = Column(Integer, default=0, nullable=False)
    completion_tokens = Column(Integer, default=0, nullable=False)
    # 缓存命中数：DeepSeek 命中价只有未命中的 1/30，不单独记就无法解释成本差异
    cache_hit_tokens = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)

    cost = Column(Numeric(18, 8), default=0, nullable=False)   # 元
    latency_ms = Column(Float, default=0, nullable=False)

    status = Column(String(20), default="ok", nullable=False, index=True)   # ok / error
    peak = Column(Boolean, default=False, nullable=False)     # 是否按峰时价计费
    error = Column(String(300))                               # status=error 时的错误摘要

    created_at = Column(DateTime, default=utc_now, nullable=False, index=True)

    # 按天 + 按节点聚合是最常用的查询，建组合索引避免全表扫
    __table_args__ = (
        Index("ix_llm_call_log_created_node", "created_at", "node"),
        # 用户视角的「按天看成本」：先按 user 缩小范围，再按时间排序
        Index("ix_llm_call_log_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<LlmCallLog(id={self.id}, node='{self.node}', "
            f"tokens={self.total_tokens}, cost={self.cost})>"
        )

# schemas.py —— 所有结构化输出模型（单一事实来源）
#
# 设计要点：把评分锚点写进 Field.description。
# JSON Schema 会作为 function calling 的参数定义「逐字段」送进模型，
# 比在 system prompt 里堆文字注意力更集中，直接对抗打分往中间挤的
# central tendency bias（原系统 0.44-0.64 的堆积就是这么来的）。
from enum import Enum
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Dimension(str, Enum):
    """固定维度枚举：下游可按维度聚合，而自由文本做不到。"""
    INNOVATION = "创新性与学术价值"
    STRUCTURE = "逻辑结构与论证"
    COMPLETENESS = "内容完整度与充分性"
    LANGUAGE = "语言表达与学术规范"
    CITATION = "文献引用质量"


class Severity(str, Enum):
    BLOCKING = "blocking"   # 必须改，否则不放行
    MAJOR = "major"
    MINOR = "minor"


class DimensionScore(BaseModel):
    dimension: Dimension = Field(description="维度名称，必须从枚举中选择")
    score: float = Field(
        ge=0.0, le=1.0,
        description=(
            "该维度得分。请按以下锚点校准，不要在 0.5-0.6 之间堆积：\n"
            "0.90-1.00 达到可发表水平；0.75-0.89 小修后可投稿；"
            "0.60-0.74 需较大修改；0.40-0.59 结构或论证存在明显缺陷；"
            "0.00-0.39 基本不成立。"
        ),
    )
    comment: str = Field(description="一句话点评，必须指向具体段落或具体缺陷，禁止空泛评价")
    evidence: str = Field(
        default="",
        description="支撑该评分的原文片段（逐字摘录 15-40 字）。无法摘录时留空字符串。",
    )


class SectionFeedback(BaseModel):
    section: str = Field(
        description="章节标题，必须逐字抄写正文中的章标题（去掉 ## 前缀）",
    )
    advice: str = Field(description="只针对该章节自身的问题，不要写全篇通用建议")
    severity: Severity = Field(default=Severity.MAJOR, description="该问题的严重程度")
    evidence: str = Field(default="", description="该章节中存在问题的原文片段")

    @field_validator("section")
    @classmethod
    def _strip(cls, v: str) -> str:
        # 方案 A：只做最基础的清洗，复杂匹配交给调用方（_norm_section_name 兜底）
        return (v or "").strip().lstrip("#").strip()


class ReviewResult(BaseModel):
    score: float = Field(
        ge=0.0, le=1.0,
        description=(
            "综合质量分，必须等于 dimensions 中 5 项得分的算术平均（保留两位）。"
            "锚点：0.90+ 可发表 / 0.75 小修 / 0.60 大修 / 0.40 结构缺陷 / 0.20 不可用。"
        ),
    )
    dimensions: List[DimensionScore] = Field(description="必须恰好 5 项，每个维度一项，不得重复")
    section_feedback: List[SectionFeedback] = Field(default_factory=list)
    overall: str = Field(description="总体评价，一段话，指出最关键的 1-2 个问题")
    must_fix: List[str] = Field(
        default_factory=list,
        description="不放行就不能通过的问题清单（只列 blocking 级），用于驱动下一轮重写",
    )

    @model_validator(mode="before")
    @classmethod
    def _tolerate_shape_drift(cls, data):
        """宽容解析：模型给的形状与 Schema 不完全一致时尽量救回，而不是整篇降级。

        实测（2026-11-01 评测）模型常给出两种漂移，信息完全等价、不应判死：
          1) dimensions 写成 dict：{"创新性与学术价值": {...}, ...} —— 键就是维度名
          2) 顶层没有 score，只有各维度分 —— 用均值补齐
        无法救的情况（维度缺失/重复）仍交给 _check 报错，不编造维度。
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)

        # 0) 外层包装漂移：{"review": {...}} / {"data": {...}} 等
        if "dimensions" not in data and "score" not in data:
            for key in ("review", "data", "result", "output", "evaluation"):
                inner = data.get(key)
                if isinstance(inner, dict) and ("dimensions" in inner or "score" in inner
                                                or "overall_score" in inner):
                    data = dict(inner)
                    break

        # 0b) 字段别名：模型偶尔用 overall_score / total_score 代替 score，
        #     或用 summary / conclusion 代替 overall（实测出现过 review.overall_score + summary）
        if "score" not in data:
            for alias in ("overall_score", "total_score", "final_score"):
                if isinstance(data.get(alias), (int, float)):
                    data["score"] = data[alias]
                    break
        if not str(data.get("overall") or "").strip():
            for alias in ("summary", "overall_comment", "conclusion", "overall_evaluation"):
                if isinstance(data.get(alias), str) and data[alias].strip():
                    data["overall"] = data[alias]
                    break

        dims = data.get("dimensions")
        if isinstance(dims, dict):
            # {"维度名": {...}} → [{"dimension": 维度名, ...}]
            converted = []
            for name, payload in dims.items():
                item = dict(payload) if isinstance(payload, dict) else {"comment": str(payload)}
                item.setdefault("dimension", name)   # 键即维度名，优先用键
                converted.append(item)
            data["dimensions"] = converted
        elif isinstance(dims, list):
            # 统一转成 dict 副本：既避免污染调用方传入的对象，
            # 也让「已经是 DimensionScore 实例」的入参能被后续量纲检查看到分数
            data["dimensions"] = [
                d.model_dump() if isinstance(d, BaseModel)
                else (dict(d) if isinstance(d, dict) else d)
                for d in dims
            ]

        # 0c) 分制漂移：模型偶尔按 10 分制打分（实测出现 score=2.2、overall_score=6.5）。
        # 判定条件严格：**所有**分数都 >1 才算整体换了量纲（量纲差 10 倍不改变相对排序）；
        # 若只有个别字段越界（如维度都是 0.8 而 score=1.7），那是垃圾输出，仍应拒绝，
        # 不能靠除以 10 把垃圾"洗"成合法值。
        _numeric = []
        if isinstance(data.get("score"), (int, float)):
            _numeric.append(float(data["score"]))
        for _d in data.get("dimensions") or []:
            if isinstance(_d, dict) and isinstance(_d.get("score"), (int, float)):
                _numeric.append(float(_d["score"]))
        if _numeric and min(_numeric) > 1.0 and max(_numeric) <= 10.0:
            if isinstance(data.get("score"), (int, float)):
                data["score"] = round(float(data["score"]) / 10.0, 3)
            for _d in data.get("dimensions") or []:
                if isinstance(_d, dict) and isinstance(_d.get("score"), (int, float)):
                    _d["score"] = round(float(_d["score"]) / 10.0, 3)

        # score 缺失/不可解析 → 用各维度均值补齐（与 _check 的一致性口径相同）
        raw_score = data.get("score")
        if raw_score is None or isinstance(raw_score, str):
            scores = [
                d.get("score") for d in (data.get("dimensions") or [])
                if isinstance(d, dict) and isinstance(d.get("score"), (int, float))
            ]
            if scores:
                data["score"] = round(sum(scores) / len(scores), 2)
        return data

    @model_validator(mode="after")
    def _check(self):
        # 1) 维度必须齐、不重
        dims = [d.dimension for d in self.dimensions]
        if len(dims) != 5 or len(set(dims)) != 5:
            raise ValueError(f"dimensions 必须恰好 5 项且不重复，实际 {dims}")

        # 2) 综合分与维度均值一致性（容差 0.03，防模型心算漂移）
        avg = sum(d.score for d in self.dimensions) / 5
        if abs(avg - self.score) > 0.03:
            self.score = round(avg, 2)   # 以维度为准，静默纠正而非报错
        return self


# ---------- 大纲 ----------

class OutlineSection(BaseModel):
    title: str = Field(description="章节标题，不含「第X章」等编号前缀")
    points: List[str] = Field(default_factory=list, description="该章要点，2-4 条")
    word_count: int = Field(default=800, ge=100, le=5000, description="该章目标字数")


class OutlineResult(BaseModel):
    title: str = Field(description="论文标题")
    sections: List[OutlineSection] = Field(
        min_length=1,
        description="章节列表，必须至少包含 1 章（摘要/引言/正文/结论等）",
    )


# ---------- 引文查证（工具的裁决格式） ----------

class CitationVerdict(BaseModel):
    verdict: Literal["supported", "partial", "unsupported", "not_found"] = Field(
        description="裁决结果：supported=文献支持该论断；partial=部分支持；"
                    "unsupported=文献与论断无关；not_found=编号越界或无正文"
    )
    similarity: float = Field(ge=0.0, le=1.0, description="论断与文献正文的语义相似度")
    reason: str = Field(description="一句话裁决理由")


# state.py —— 论文生成工作流的状态定义
from typing import TypedDict, List, Optional

class PaperState(TypedDict, total=False):
    """
    论文生成工作流状态（节点之间传递的“共享黑板”）。

    每个节点从黑板上读材料、把结果写回黑板。
    total=False 表示所有字段都是可选的——初始状态只需传少数几个字段即可启动。

    与文档原版的区别：
    1. sections_content 不再用 add_messages 合并器（那是给聊天消息用的，套在 dict 上会出错）
    2. 补上了 analysis_result 字段（原版节点返回了它却没在状态里定义，LangGraph 会报错）
    3. 新增 topic_recommended 字段，用于给选题推荐循环“装刹车”
    """

    # ---- 基本信息 ----
    paper_id: Optional[int]
    user_id: Optional[int]            # 论文作者：成本流水按用户归因用（不靠 papers 反查）
    title: str
    topic: str
    keywords: List[str]

    # ---- 工作流控制 ----
    current_step: str              # 当前走到哪一步（供前端显示进度）
    is_topic_clear: bool           # 选题是否明确
    topic_recommended: bool        # 是否已经推荐过选题（防死循环的刹车）
    requires_revision: bool

    # ---- 内容存储 ----
    analysis_result: Optional[str]     # 选题分析/推荐结果
    outline: Optional[dict]            # 大纲（结构化 dict）
    sections_content: dict             # 各章节正文 {章节标题: 内容}

    # ---- 文献相关 ----
    references: List[dict]             # 检索到的文献列表（写作前检索，供写作与文末参考文献共用）
    references_text: Optional[str]     # 编号化的参考文献文本（与正文 [n] 编号同源同序）
    formatted_references: Optional[str]  # 格式化后的参考文献文本

    # ---- 质量控制 ----
    quality_score: Optional[float]   # None = 评审降级（不编分数）
    quality_degraded: bool           # True = 本轮评审不可用，跳过质量门禁放行
    quality_comment: Optional[str]      # 评审意见（LLM 评审专家输出）
    quality_dimensions: List[dict]      # 各维度评分明细
    quality_section_feedback: List[dict]  # 分章节改进建议 [{section, advice}]，供重写时逐章下发
    revision_count: int
    max_revisions: int
    # 首次质量检查的分数（重写前的原始分数）。用途：标定质量阈值 ——
    # 只有知道「未重写时能拿多少分」，才能判断阈值该定在哪（否则阈值只能靠猜，
    # 2026-11-01 实测就卡在这里：最终分 0.44-0.53，但首检分无记录）。
    first_pass_score: Optional[float]

    # ---- 历史最优版本（best-so-far）----
    # 评审打分本身有抖动，重写后分数可能反而更低。这里记录「截至目前得分最高」
    # 的那一版正文，保存时优先用它，避免把更好的一版丢掉。
    best_score: Optional[float]
    best_paper: Optional[str]
    best_comment: Optional[str]
    best_dimensions: List[dict]
    best_sections: dict

    # ---- 输出 ----
    final_paper: Optional[str]


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
    references: List[dict]             # 检索到的文献列表
    formatted_references: Optional[str]  # 格式化后的参考文献文本

    # ---- 质量控制 ----
    quality_score: float
    quality_comment: Optional[str]      # 评审意见（LLM 评审专家输出）
    quality_dimensions: List[dict]      # 各维度评分明细
    revision_count: int
    max_revisions: int

    # ---- 输出 ----
    final_paper: Optional[str]

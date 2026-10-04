
# graph_builder.py —— LangGraph 工作流图构建器
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.workflows.state import PaperState
from app.agents.topic_agent import TopicAgent
from app.agents.outline_agent import OutlineAgent
from app.agents.writer_agent import WriterAgent
from app.agents.reference_agent import ReferenceAgent
from app.agents.polish_agent import PolishAgent
from app.agents.reviewer_agent import ReviewerAgent

class PaperWorkflowGraph:
    """
    论文生成工作流图构建器。

    流水线形状：
      选题分析 →(条件)→ 大纲生成 → 检索文献 → 章节撰写 → 文献整理 → 全文润色 → 质量检查
      质量检查 →(条件)→ 结束；不合格则回到“章节撰写”重写（有次数上限）

    2026-10-04 关键修复：
    RAG 检索结果原先排在大纲之后、写作之后（add_references），写作时拿不到文献，
    writer_agent 的 references 入参永远是空的（走“暂无参考文献”兜底），
    RAG 对正文零贡献 —— 这也解释了消融实验里 no_rag 反而 ≥ full。
    现在把检索前移到写作之前（retrieve_references），写作节点从 state 取文献注入 Prompt；
    add_references 改为“复用已检索结果”，不再重复检索、也不再重复发 LLM 请求。

    与文档原版的区别：
    1. 删掉了 SqliteSaver 导入（没装那个包，原版也没用到，导入即崩）
    2. 选题推荐循环加了“最多一次”的刹车，防止死循环
    3. 节点不再直接篡改 state，而是复制一份、改完再返回（LangGraph 最佳实践）
    """

    def __init__(self, checkpointer=None):
        # 初始化 6 个 Agent（流水线上的 6 位员工）
        self.topic_agent = TopicAgent()
        self.outline_agent = OutlineAgent()
        self.writer_agent = WriterAgent()
        self.reference_agent = ReferenceAgent()
        self.polish_agent = PolishAgent()
        self.reviewer_agent = ReviewerAgent()

        # checkpoint 断点续跑：默认启用内存版（进程内可恢复），
        # 也可传入持久化 saver（如 SqliteSaver）实现跨进程断点续跑
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()

        # 构建并编译工作流图
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建工作流图"""
        workflow = StateGraph(PaperState)

        # 1) 注册节点：节点名 → 对应的处理函数
        workflow.add_node("analyze_topic", self._analyze_topic)
        workflow.add_node("generate_outline", self._generate_outline)
        workflow.add_node("retrieve_references", self._retrieve_references)
        workflow.add_node("write_sections", self._write_sections)
        workflow.add_node("add_references", self._add_references)
        workflow.add_node("polish_paper", self._polish_paper)
        workflow.add_node("quality_check", self._quality_check)

        # 2) 入口：从选题分析开始
        workflow.add_edge(START, "analyze_topic")

        # 3) 条件边①：选题明确 → 生成大纲；不明确 → 再推荐一轮（最多一轮）
        workflow.add_conditional_edges(
            "analyze_topic",
            self._should_recommend_topic,
            {
                "recommend": "analyze_topic",
                "outline": "generate_outline",
            },
        )

        # 4) 普通边：固定顺序往下走
        #    关键：检索必须排在写作之前，写作时才有文献可注入
        workflow.add_edge("generate_outline", "retrieve_references")
        workflow.add_edge("retrieve_references", "write_sections")
        workflow.add_edge("write_sections", "add_references")
        workflow.add_edge("add_references", "polish_paper")
        workflow.add_edge("polish_paper", "quality_check")

        # 5) 条件边②：质量达标 → 结束；不达标 → 回到章节撰写重写
        workflow.add_conditional_edges(
            "quality_check",
            self._check_quality,
            {
                "pass": END,
                "revise": "write_sections",
            },
        )

        return workflow.compile(checkpointer=self.checkpointer)

    # ---------- 以下是各节点的处理函数（每个对应一个工位）----------

    async def _analyze_topic(self, state: PaperState) -> dict:
        """节点1：选题分析。选题明确就直接往下；不明确就请选题Agent推荐"""
        topic = (state.get("topic") or "").strip()
        keywords = state.get("keywords") or []

        if topic:
            # 选题明确：不用推荐，直接进下一步
            return {
                "current_step": "topic_analysis",
                "is_topic_clear": True,
                "analysis_result": f"选题明确：{topic}",
            }

        # 选题不明确：调用选题Agent推荐选题
        result = await self.topic_agent.execute({
            "field": "用户感兴趣的研究方向",
            "keywords": ", ".join(keywords),
        })
        return {
            "current_step": "topic_analysis",
            "is_topic_clear": False,
            "topic_recommended": True,   # 标记“已推荐过一次”，刹车生效
            "analysis_result": result.get("topics", ""),
        }

    async def _generate_outline(self, state: PaperState) -> dict:
        """节点2：生成论文大纲"""
        result = await self.outline_agent.execute({
            "title": state.get("title", ""),
            "paper_type": "research",
            "word_limit": 6000,
        })
        return {
            "current_step": "outline",
            "outline": result.get("outline_dict") or {},
        }

    async def _retrieve_references(self, state: PaperState) -> dict:
        """节点3：写作前检索文献（RAG 的知识供给环节）。

        全篇检索一次，各章节共用同一批文献 —— 与既有评测口径（top_k=5、每篇 5 条参考文献）保持一致，
        便于新旧评测结果直接对比。只做向量检索、不调 LLM，因此几乎不增加耗时。
        """
        query = state.get("title") or state.get("topic") or ""
        references = await self.reference_agent.retrieve(query, top_k=5)
        return {
            "current_step": "references",
            "references": references,
            # 编号化文本在这里算好：正文引用与文末参考文献共用同一份顺序，编号不会错位
            "references_text": self.reference_agent.format_gbt(references),
        }

    async def _write_sections(self, state: PaperState) -> dict:
        """节点4：按大纲逐章撰写（把检索到的文献注入写作 Prompt）"""
        outline = state.get("outline") or {}
        sections = outline.get("sections") or []

        # 兜底：万一大纲里没有章节列表（LLM输出格式不稳定），就整篇当一节写
        if not sections:
            sections = [{"title": state.get("title", "正文"), "points": [], "word_count": 800}]

        # 复制一份再改，不直接篡改共享状态（LangGraph 最佳实践）
        sections_content = dict(state.get("sections_content") or {})

        # 若上一轮质量检查给了改进意见（打回重写），把它喂给写作专家，让重写有的放矢
        feedback = state.get("quality_comment", "")

        # 写作前的检索结果 → 渲染成带编号的引用字符串列表，逐章注入
        # 列表下标 i 对应正文里的 [i+1]，与文末 references_text 同源同序
        references = self.reference_agent.format_citation_strings(
            state.get("references") or []
        )

        for section in sections:
            title = section.get("title", "")
            result = await self.writer_agent.execute({
                "section_title": title,
                "outline_points": section.get("points", []),
                "word_count": section.get("word_count", 800),
                "feedback": feedback,
                "references": references,          # ← 修复点：原先漏传，写作时永远「暂无参考文献」
            })
            sections_content[title] = result.get("content", "")

        return {
            "current_step": "writing",
            "sections_content": sections_content,
        }

    async def _add_references(self, state: PaperState) -> dict:
        """节点5：整理参考文献（复用检索结果，不再重复检索）。

        前置节点已完成检索，这里只负责把文献渲染成 GB/T 7714 编号列表。
        若前置检索结果为空（例如单独调用本节点、或向量库为空），
        退回原来的「检索 + LLM 整理」路径做兼容兜底。
        """
        references = state.get("references") or []

        if references:
            return {
                "current_step": "references",
                "formatted_references": self.reference_agent.format_gbt(references),
            }

        result = await self.reference_agent.execute({
            "query": state.get("title", ""),
            "top_k": 5,
        })
        return {
            "current_step": "references",
            "references": result.get("references", []),
            "formatted_references": result.get("formatted", ""),
        }

    async def _polish_paper(self, state: PaperState) -> dict:
        """节点6：把各章拼成全文并润色"""
        full_content = "\n\n".join([
            f"## {title}\n\n{content}"
            for title, content in (state.get("sections_content") or {}).items()
        ])

        result = await self.polish_agent.execute({
            "content": full_content,
            "focus": "all",
        })
        return {
            "current_step": "polish",
            "final_paper": result.get("polished", ""),
        }

    async def _quality_check(self, state: PaperState) -> dict:
        """
        节点7：质量检查（LLM 评审版）。
        调用 ReviewerAgent 让大模型从 5 个维度打分并给改进建议；
        解析失败或调用异常时，退回旧的启发式评分（有正文+有文献=0.8），保证流程不崩。
        """
        revision_count = state.get("revision_count", 0)

        # 兜底：启发式评分（与旧版一致）
        def fallback_score(reason: str):
            has_content = bool(state.get("final_paper"))
            has_refs = bool(state.get("references"))
            return 0.8 if (has_content and has_refs) else 0.5, reason, []

        try:
            result = await self.reviewer_agent.execute({
                "title": state.get("title", ""),
                "topic": state.get("topic", ""),
                "content": state.get("final_paper", ""),
                "references": state.get("formatted_references", ""),
            })
            data = result.get("data") or {}
            raw_score = data.get("score")
            if raw_score is None:
                score, comment, dimensions = fallback_score("（评审结果解析失败，使用基础评分）")
            else:
                score = max(0.0, min(1.0, float(raw_score)))
                comment = data.get("overall", "") or ""
                dimensions = data.get("dimensions", []) or []
        except Exception as e:
            score, comment, dimensions = fallback_score(f"（评审服务异常，使用基础评分：{e}）")

        return {
            "current_step": "quality_check",
            "quality_score": score,
            "quality_comment": comment,
            "quality_dimensions": dimensions,
            "revision_count": revision_count + 1,
        }

    # ---------- 以下是两个条件路由函数（决定走哪条边）----------

    def _should_recommend_topic(self, state: PaperState) -> str:
        """选题分析后的路由：明确→大纲；不明确但已推荐过→也放行（防死循环）；否则→再推荐"""
        if state.get("is_topic_clear"):
            return "outline"
        if state.get("topic_recommended"):
            return "outline"
        return "recommend"

    def _check_quality(self, state: PaperState) -> str:
        """质量检查的路由：达标或通过次数用尽 → 结束；否则 → 打回重写"""
        if state.get("quality_score", 0) >= 0.7:
            return "pass"
        if state.get("revision_count", 0) >= state.get("max_revisions", 3):
            return "pass"
        return "revise"

    # ---------- 对外接口 ----------

    @staticmethod
    def _thread_config(config, state):
        """没传 config 时自动生成一个线程 id：每篇论文一个独立 checkpoint 线程"""
        if config is not None:
            return config
        thread_id = f"paper-{state.get('paper_id') or 'anon'}"
        return {"configurable": {"thread_id": thread_id}}

    async def run(self, initial_state: PaperState, config: Optional[dict] = None):
        """运行工作流，返回最终状态。
        config 可传 {"configurable": {"thread_id": ...}}；
        同一 thread_id + 传 None 输入可断点续跑（配合 checkpointer）。"""
        cfg = self._thread_config(config, initial_state)
        return await self.graph.ainvoke(initial_state, config=cfg)

    async def stream(self, initial_state: PaperState, config: Optional[dict] = None):
        """流式运行：逐个节点产出结果（用于显示进度）"""
        cfg = self._thread_config(config, initial_state)
        async for event in self.graph.astream(initial_state, config=cfg):
            yield event

# 全局工作流实例（import 后直接可用）
paper_workflow = PaperWorkflowGraph()

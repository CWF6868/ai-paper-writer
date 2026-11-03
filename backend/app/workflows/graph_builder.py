
# graph_builder.py —— LangGraph 工作流图构建器
import asyncio
import logging
import re
import time
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.workflows.state import PaperState
from app.config import settings
from app.agents.topic_agent import TopicAgent
from app.agents.outline_agent import OutlineAgent
from app.agents.writer_agent import WriterAgent
from app.agents.reference_agent import ReferenceAgent
from app.agents.polish_agent import PolishAgent
from app.agents.reviewer_agent import ReviewerAgent
from app.llm.structured import StructuredOutputError
from app.llm.tools.context import set_current_references
from app.core.llm_client import llm_scope

logger = logging.getLogger(__name__)

def _norm_section_name(name: str) -> str:
    """把「第3章 实验结果」「三、实验结果」「3. 实验结果」统一成「实验结果」。"""
    s = str(name or "").strip().lstrip("#").strip()
    s = re.sub(r"^(第\s*[一二三四五六七八九十百0-9]+\s*[章节部分篇]|[0-9]+\s*[\.、）)]|[一二三四五六七八九十]+\s*[、\.）)])\s*", "", s)
    return s.strip()


def _pick_section_advice(section_title: str, items) -> str:
    """
    从评审给出的「分章节建议」里挑出属于当前章节的那一条。

    评审输出的章节名常与大纲标题不一致（多/少「第X章」「一、」「3.」等前缀），
    因此先归一化再判定「相等 / 互相包含」；匹配不到返回空串，由调用方退回整篇意见。
    """
    title = _norm_section_name(section_title)
    if not title:
        return ""
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = _norm_section_name(item.get("section"))
        advice = str(item.get("advice") or "").strip()
        if not name or not advice:
            continue
        if name == title or name in title or title in name:
            return advice
    return ""


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

    @staticmethod
    def _metered(node_name: str, fn):
        """给节点包一层 LLM 计量作用域：该节点内所有模型调用都归到 node_name 下。

        用包装器而不是在每个业务方法里写 with：不动缩进、不污染业务逻辑，
        新增节点时也不会忘记标记（漏标 = 成本表里出现 unknown 节点）。
        """
        async def _wrapped(state: PaperState) -> dict:
            with llm_scope(
                node=node_name,
                paper_id=state.get("paper_id"),
                user_id=state.get("user_id"),
            ):
                return await fn(state)

        return _wrapped

    def _build_graph(self):
        """构建工作流图"""
        workflow = StateGraph(PaperState)

        # 1) 注册节点：节点名 → 对应的处理函数
        # 用 _metered 包一层：给每个节点内的所有模型调用打上 node/paper_id 标签，
        # 成本才能按节点归因。不改业务方法体，避免把缩进和逻辑一起搅乱。
        workflow.add_node("analyze_topic", self._metered("analyze_topic", self._analyze_topic))
        workflow.add_node("generate_outline", self._metered("generate_outline", self._generate_outline))
        workflow.add_node("retrieve_references", self._metered("retrieve_references", self._retrieve_references))
        workflow.add_node("write_sections", self._metered("write_sections", self._write_sections))
        workflow.add_node("add_references", self._metered("add_references", self._add_references))
        workflow.add_node("polish_paper", self._metered("polish_paper", self._polish_paper))
        workflow.add_node("quality_check", self._metered("quality_check", self._quality_check))

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
        """节点2：生成论文大纲（Schema 结构化输出）。

        大纲 Agent 三级降级全失败时不再拿 {"raw": ...} 假装是大纲：
        返回空 outline，由 _write_sections 的「整篇当一节写」兜底接住。
        """
        try:
            result = await self.outline_agent.execute({
                "title": state.get("title", ""),
                "paper_type": "research",
                "word_limit": 6000,
            })
            outline_dict = result.get("outline_dict") or {}
        except StructuredOutputError as e:
            # 原因必须写进 message：默认 Formatter 不渲染 extra，写 extra 等于丢弃
            logger.error("outline degraded (node=generate_outline): %s", e)
            outline_dict = {}
        return {
            "current_step": "outline",
            "outline": outline_dict,
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
        """节点4：按大纲逐章撰写（受控并发 + 保序回填）。

        2026-11-02 P0-2：由串行 for-await 改为 asyncio.gather 受控并发。
        章节写作是纯网络等待（单章 6-10s），串行写 8 章 = 8 倍等待，
        而章节之间没有数据依赖（各章拿同一批文献、各写自己的要点），
        因此可以并发；用 Semaphore 限流，避免一次把 8 个请求全砸给 DeepSeek。

        两个必须守住的约束：
        1. **章节顺序**：sections_content 的插入序就是成文序（_polish_paper 按它拼全文，
           _save_results 按它生成 order_index）。并发完成顺序是不确定的，
           因此结果按大纲下标回填，绝不能"谁先写完谁在前"。
        2. **失败可定位**：gather 用 return_exceptions=True 收齐所有结果，
           报错时指明是哪一章、什么异常 —— 裸异常只告诉你"某处失败"，
           并发下等于放弃排查。
        """
        outline = state.get("outline") or {}
        sections = outline.get("sections") or []

        # 兜底：万一大纲里没有章节列表（LLM输出格式不稳定），就整篇当一节写
        if not sections:
            sections = [{"title": state.get("title", "正文"), "points": [], "word_count": 800}]

        # 复制一份再改，不直接篡改共享状态（LangGraph 最佳实践）
        sections_content = dict(state.get("sections_content") or {})

        # 上一轮质量检查给了改进意见（打回重写）：优先用「分章节建议」，
        # 只有该章节匹配不到专属建议时，才退回整篇意见。
        global_feedback = state.get("quality_comment", "")
        section_feedback = state.get("quality_section_feedback") or []

        # 写作前的检索结果 → 渲染成带编号的引用字符串列表，逐章注入
        # 列表下标 i 对应正文里的 [i+1]，与文末 references_text 同源同序
        references = self.reference_agent.format_citation_strings(
            state.get("references") or []
        )

        # 每次写作轮次一个信号量：限的是「同一篇论文同一轮内的在途请求数」。
        # 多篇论文同时生成时总在途 ≈ 篇数 × 该值，需要时按账号配额调小。
        concurrency = max(1, int(settings.SECTION_CONCURRENCY or 1))
        sem = asyncio.Semaphore(concurrency)
        durations: list[float] = []   # 每章实际调用耗时：用来在日志里对账"串行本应多久"

        async def _write_one(section: dict) -> str:
            title = section.get("title", "")
            # 分章建议的匹配是纯字符串运算，放在信号量外，不占并发额度
            advice = _pick_section_advice(title, section_feedback)
            # 计时器必须在信号量**里面**：在外面会把排队等额度的时间也算成"调用耗时"，
            # 算出 6x 这种超过并发上限的假加速（并发上限 4，加速不可能 > 4）
            async with sem:
                t_call = time.perf_counter()
                try:
                    result = await self.writer_agent.execute({
                        # 论文题目必须传：否则每章只能拿到章节名，写出来是同质化的通用段落
                        "paper_title": state.get("title", ""),
                        "section_title": title,
                        "outline_points": section.get("points", []),
                        "word_count": section.get("word_count", 800),
                        "feedback": advice or global_feedback,
                        "references": references,      # ← 修复点：原先漏传，写作时永远「暂无参考文献」
                    })
                finally:
                    durations.append(time.perf_counter() - t_call)   # 失败也计入，别让分子漏掉
            return result.get("content", "") or ""

        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[_write_one(s) for s in sections], return_exceptions=True
        )
        elapsed = time.perf_counter() - t0

        # 取消必须原样上抛，不能包成普通异常（否则外层无法区分"用户断开"和"章节写挂了"）
        for r in results:
            if isinstance(r, asyncio.CancelledError):
                raise r

        failures = [
            (sections[i].get("title", ""), r)
            for i, r in enumerate(results)
            if isinstance(r, BaseException)
        ]
        if failures:
            detail = "；".join(f"「{t}」{type(e).__name__}: {e}" for t, e in failures)
            logger.error(
                "write_sections failed: %d/%d sections, concurrency=%d, %.1fs — %s",
                len(failures), len(sections), concurrency, elapsed, detail,
            )
            raise RuntimeError(
                f"章节写作失败（{len(failures)}/{len(sections)} 章）：{detail}"
            ) from failures[0][1]

        # 按大纲顺序回填（不是按完成顺序）：这是成文顺序的唯一保证
        for section, content in zip(sections, results):
            sections_content[section.get("title", "")] = content

        logger.info(
            "write_sections done: %d sections in %.1fs (concurrency=%d, 串行本应 %.1fs, 加速 %.2fx)",
            len(sections), elapsed, concurrency, sum(durations),
            (sum(durations) / elapsed) if elapsed else 0,
        )

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
        节点7：质量检查（LLM 评审版，Schema 结构化输出 + 工具调用）。

        评审不可用（异常 / 三级降级全失败 / 分数越界）时，不再编造 0.8 假分数：
        quality_score=None + quality_degraded=True，放行但不假装合格。
        假分数会进库、进评测统计，是数据污染——比功能缺失严重得多。
        """
        revision_count = state.get("revision_count", 0)

        def _degraded(reason: str) -> dict:
            """显式降级：不编分数、不动历史最优、不塞脏建议。"""
            return {
                "current_step": "quality_check",
                "quality_score": None,
                "quality_degraded": True,
                "quality_comment": f"评审不可用（{reason}），本轮跳过质量门禁",
                "quality_dimensions": [],
                "quality_section_feedback": [],
                "revision_count": revision_count + 1,
            }

        try:
            # 工具上下文注入：verify_citation / search_literature 从这里拿文献列表，
            # 用 contextvars 保证并发评审时数据不串（不能塞进全局变量）
            with set_current_references(state.get("references") or []):
                result = await self.reviewer_agent.execute({
                    "title": state.get("title", ""),
                    "topic": state.get("topic", ""),
                    "content": state.get("final_paper", ""),
                    "references": state.get("formatted_references", ""),
                })
        except Exception as e:  # 评审服务整体异常 → 显式降级，不崩流程也不编分数
            # 原因必须写进 message：默认 Formatter 不渲染 extra，写 extra 等于丢弃
            logger.error("reviewer degraded (node=quality_check): %s", e)
            return _degraded(str(e))

        data = result.get("data") or {}
        raw_score = data.get("score")
        # 防御：任何路径漏进来的缺失分/越界分都按降级处理（Pydantic 正常已拦住，
        # 这里兜住「reviewer 被外部替换成非结构化实现」这类情况）
        if raw_score is None:
            return _degraded("评审结果缺少 score")
        try:
            score = round(float(raw_score), 2)
        except (TypeError, ValueError):
            return _degraded(f"分数不可解析: {raw_score!r}")
        if not (0.0 <= score <= 1.0):
            return _degraded(f"分数越界: {score}")

        comment = data.get("overall", "") or ""
        dimensions = data.get("dimensions", []) or []
        # 分章节建议：minor 级建议不进重写（改小问题反而破坏结构），只下发 blocking/major
        section_feedback = [
            fb for fb in (data.get("section_feedback") or [])
            if isinstance(fb, dict) and str(fb.get("severity", "major")).lower() != "minor"
        ]
        # must_fix（blocking 级清单）拼进评审意见，驱动下一轮重写有的放矢
        must_fix = [str(m) for m in (data.get("must_fix") or []) if str(m).strip()]
        if must_fix:
            comment = comment + "\n必须修复：\n" + "\n".join(f"- {m}" for m in must_fix)

        # 记录「历史最优版本」：评审分数有抖动，重写后可能反而更低，
        # 因此每轮都和历史最高分比较，只保留更好的那一版，避免最终保存的是退化版本。
        current_paper = state.get("final_paper") or ""
        prev_best = state.get("best_score")
        if prev_best is None or score > prev_best:
            best_score = score
            best_paper = current_paper
            best_comment = comment
            best_dimensions = dimensions
            best_sections = dict(state.get("sections_content") or {})
        else:
            best_score = prev_best
            best_paper = state.get("best_paper") or ""
            best_comment = state.get("best_comment") or ""
            best_dimensions = state.get("best_dimensions") or []
            best_sections = state.get("best_sections") or {}

        return {
            "current_step": "quality_check",
            "quality_score": score,
            "quality_degraded": False,
            # 首检分只记一次：后续重写轮次不覆盖，用于标定质量阈值
            "first_pass_score": (
                score if revision_count == 0 else (state.get("first_pass_score") or score)
            ),
            "quality_comment": comment,
            "quality_dimensions": dimensions,
            "quality_section_feedback": section_feedback,
            "revision_count": revision_count + 1,
            "best_score": best_score,
            "best_paper": best_paper,
            "best_comment": best_comment,
            "best_dimensions": best_dimensions,
            "best_sections": best_sections,
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
        """质量检查的路由：评审降级 → 放行但不假装合格；达标或次数用尽 → 结束；否则 → 打回重写"""
        if state.get("quality_degraded"):
            return "pass"     # 评审不可用：放行，但不编分数
        if (state.get("quality_score") or 0) >= settings.QUALITY_PASS_THRESHOLD:
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

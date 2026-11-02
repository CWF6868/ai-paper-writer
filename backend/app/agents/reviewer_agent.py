# reviewer_agent.py —— 学术评审专家 Agent（质量检查的 LLM 版）
#
# P0-1 改造：解析不再依赖 parse_json 字符串约定，出口由 ReviewResult Schema 保证；
# 工具调用（verify_citation / search_literature）走显式子图，预算硬上限。
# 返回值保持旧契约：data 仍是 dict（形状与原 parse_json 结果一致），
# 因此 graph_builder 的 data.get("score") 与既有测试断言结构不用改。
from typing import Any, Dict, Optional, Sequence

from langchain_core.prompts import ChatPromptTemplate

from app.agents.base_agent import BaseAgent
from app.agents.reviewer_graph import run_reviewer
from app.llm.tools.registry import REVIEWER_TOOLS
from app.utils.llm import get_llm


# 角色与评分纪律：收口（finalize）阶段也要用，因此独立成常量
_ROLE_PROMPT = """你是一位严谨的学术论文评审专家，请以审稿人视角评审论文，从 5 个固定维度打分：
1. 创新性与学术价值
2. 逻辑结构与论证
3. 内容完整度与充分性
4. 语言表达与学术规范
5. 文献引用质量

规则：
- section_feedback 逐章给出：section 逐字抄写正文里的章标题（去掉 ## 前缀），
  advice 只写「该章节自身」需要改什么，不要把全文意见套到每一章。
- must_fix 只列 blocking 级问题（不修就不放行的那类），没有就给空列表。"""

# 工具说明只在与工具环一起使用时出现
_TOOLS_PROMPT = """- 你可以使用工具查证：
  * verify_citation(ref_index, claim)：核验正文论断是否被第 ref_index 条参考文献支持；
  * search_literature(query, top_k)：在本文参考文献的语料正文里语义检索相关文献。
  对存疑的引用先用工具查证再打分；不要为了用工具而用工具。"""

_OUTPUT_RULE = "- 输出必须是符合评审 Schema 的结构化结果。"


class ReviewerAgent(BaseAgent):
    """以审稿人视角对论文做多维度打分，并给出可执行的改进建议"""

    def __init__(self, llm: Optional[Any] = None, tools: Optional[Sequence] = None):
        # 评审是「打分」任务，不是「创作」任务：必须用 temperature=0 保证同一篇文章
        # 得到可复现的分数。默认的 0.7 会让同一份内容在 0.36~0.54 之间随机漂移，
        # 导致「重写后分数反而更低」的错觉。
        #
        # tools：None → 默认工具箱；[] → 禁用工具（纯结构化评审）；单测可注入假工具。
        super().__init__(
            name="学术评审专家",
            description="对论文进行多维度质量评估，输出分数与改进建议",
            llm=llm or get_llm(temperature=0.0),
        )
        self.tools = tools if tools is not None else REVIEWER_TOOLS

    def get_system_prompt(self) -> str:
        # 评分锚点与字段约束在 Schema description 里（会逐字段送进模型），
        # 这里只保留角色、维度、工具用法与输出纪律。
        return f"{_ROLE_PROMPT}\n{_TOOLS_PROMPT}\n{_OUTPUT_RULE}"

    def get_finalize_system_prompt(self) -> str:
        """收口专用 prompt：**不含工具说明**。

        收口时只绑定 ReviewResult 一个结构化输出工具。若 system prompt 仍写着
        verify_citation，模型会幻觉去调它，而结构化解析器里没有这个工具 →
        KeyError('verify_citation') → function_calling 这一级白白失败
        （2026-11-01 评测实测到的降级主因之一）。
        """
        return f"{_ROLE_PROMPT}\n{_OUTPUT_RULE}"

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", """请评审下面这篇论文。

## 论文标题
{title}

## 研究方向
{topic}

## 正文内容
{content}

## 参考文献
{references}"""),
        ])
        messages = prompt.format_messages(
            title=input_data.get("title", ""),
            topic=input_data.get("topic", ""),
            content=input_data.get("content", ""),
            references=input_data.get("references", ""),
        )

        # 评审子图：agent → tools（预算硬上限）→ finalize 结构化收口
        # finalize_system_prompt：收口时替换掉带工具说明的 system prompt，防止幻觉调工具
        result, mode = await run_reviewer(
            self.llm, messages, tools=self.tools,
            finalize_system_prompt=self.get_finalize_system_prompt(),
        )

        # 兼容旧字段：raw 是 JSON 文本，data 是 dict（形状与旧 parse_json 结果一致）
        # 注意：mode='json' 让 Dimension/Severity 枚举序列化成中文值（创新性与学术价值），
        # 与旧契约「dimension 是纯字符串」保持一致；默认 model_dump 会输出 Dimension.INNOVATION。
        return {
            "raw": result.model_dump_json(),
            "data": result.model_dump(mode="json"),
            "degraded": mode != "function_calling",   # 降级到 json_mode/raw_parse 即显式标记
        }

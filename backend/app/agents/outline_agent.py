# outline_agent.py —— 大纲专家 Agent
#
# P0-1 改造：parse_json 字符串约定 → OutlineResult Schema 结构化输出。
# 返回值保持旧契约：outline_dict 仍是 dict（键形状不变），outline 是 JSON 文本。
# 解析/校验失败抛 StructuredOutputError，由 graph_builder 显式降级
# （整篇当一节写的兜底逻辑在 _write_sections，那里已有）。
from langchain_core.prompts import ChatPromptTemplate

from app.agents.base_agent import BaseAgent
from app.llm.schemas import OutlineResult
from app.llm.structured import call_structured


class OutlineAgent(BaseAgent):
    """大纲专家：根据论文标题生成结构化的论文大纲"""

    def __init__(self):
        super().__init__(
            name="大纲专家",
            description="根据论文标题生成结构化的论文大纲",
        )

    def get_system_prompt(self) -> str:
        # 字段约束（章节要点数量、字数范围等）在 OutlineResult Schema description 里。
        return """你是一位资深学术论文大纲专家，擅长根据标题设计清晰、有逻辑的论文结构。

要求：
1. 大纲需包含：摘要、引言、正文若干章、结论
2. 每章给出 2-4 个要点
3. 为每章合理分配字数（总字数按用户要求）
4. 输出符合大纲 Schema 的结构化结果。"""

    async def execute(self, inputs: dict) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", "请为下面的论文设计大纲：\n标题：{title}\n论文类型：{paper_type}\n总字数要求：约 {word_limit} 字"),
        ])
        messages = prompt.format_messages(
            title=inputs.get("title", ""),
            paper_type=inputs.get("paper_type", "research"),
            word_limit=inputs.get("word_limit", 6000),
        )

        result, mode = await call_structured(self.llm, OutlineResult, messages, node="outline")

        # 兼容旧字段：outline_dict 是 dict（sections 形状不变），outline 是 JSON 文本
        return {
            "outline": result.model_dump_json(),
            "outline_dict": result.model_dump(mode="json"),
            "degraded": mode != "function_calling",
        }

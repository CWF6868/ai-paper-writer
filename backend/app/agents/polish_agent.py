# polish_agent.py —— 润色优化 Agent（第 5 个）
from typing import Dict, Any, AsyncIterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.agents.base_agent import BaseAgent


class PolishAgent(BaseAgent):
    """对论文进行语言、逻辑、规范三个维度的润色优化"""

    def __init__(self):
        super().__init__(
            name="润色专家",
            description="优化论文语言表达和逻辑结构",
        )

    def get_system_prompt(self) -> str:
        return """你是一位资深的学术论文润色专家，擅长优化论文的语言表达和逻辑结构。

润色要点：
1. **语言优化**：消除冗余表达、提升准确性、统一术语、优化句式
2. **逻辑优化**：强化论证链条、完善段落衔接、提升连贯性、优化层次
3. **规范检查**：学术用语规范、格式排版规范、引用格式规范

请保持原文学术观点，仅做表达和结构优化。"""

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        content = input_data.get("content", "")
        focus = input_data.get("focus", "all")

        # 根据 focus 参数给大模型不同的侧重点
        focus_instructions = {
            "language": "请重点优化语言表达，提升表达的准确性和简洁性。",
            "logic": "请重点优化逻辑结构，强化论证链条和段落衔接。",
            "format": "请重点检查格式规范，确保符合学术写作规范。",
            "all": "请全面优化语言表达、逻辑结构和格式规范。",
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", """请润色以下论文内容：

{focus_instruction}

## 原文内容：

{content}

## 输出要求：

只输出润色后的完整论文正文，不要附加任何修改说明、注释、前言或额外文字。

请开始润色："""),
        ])
        chain = prompt | self.llm | StrOutputParser()

        result = await chain.ainvoke({
            "content": content,
            "focus_instruction": focus_instructions.get(focus, focus_instructions["all"]),
        })

        return {
            "polished": result,
            "original_length": len(content),
            "polished_length": len(result),
        }

    async def stream_execute(self, input_data: Dict[str, Any]) -> AsyncIterator[str]:
        content = input_data.get("content", "")
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", f"请润色以下论文内容：\n\n{content}"),
        ])
        chain = prompt | self.llm | StrOutputParser()

        async for chunk in chain.astream({}):
            yield chunk
# writer_agent.py —— 内容撰写 Agent（第 3 个）
from typing import Dict, Any, AsyncIterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.agents.base_agent import BaseAgent


class WriterAgent(BaseAgent):
    """根据大纲中的某一章节要点，撰写高质量学术内容"""

    def __init__(self):
        super().__init__(
            name="写作专家",
            description="撰写高质量学术论文内容",
        )

    def get_system_prompt(self) -> str:
        return """你是一位资深的学术论文写作专家，擅长撰写高质量、逻辑严谨的学术论文内容。

写作要求：
1. 语言规范，表达准确，符合学术写作规范
2. 论点清晰，论据充分，逻辑严密
3. 合理引用参考文献，使用标准引用格式
4. 专业术语使用恰当

写作风格：
- 客观中立，避免主观判断
- 数据驱动，有理有据
- 层次清晰，结构合理
- 语言精练，避免冗余"""

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        section_title = input_data.get("section_title", "")
        outline_points = input_data.get("outline_points", [])
        references = input_data.get("references", [])
        style = input_data.get("style", "formal")
        word_count = input_data.get("word_count", 1000)
        feedback = input_data.get("feedback", "")

        # 把要点列表拼成一行行文本，更容易让模型理解
        points_text = "\n".join(f"- {p}" for p in outline_points)
        # 文献列表：有则编号排列，无则提示暂无
        refs_text = (
            "\n".join(f"[{i+1}] {ref}" for i, ref in enumerate(references))
            if references
            else "暂无参考文献"
        )
        # 上一轮的评审改进意见（首次撰写为空）
        feedback_text = feedback or "（首次撰写，暂无历史意见）"

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", """请撰写以下章节内容：

## 章节标题：{section_title}

### 内容要点：
{points}

### 相关文献：
{references}

### 历史评审改进建议：
{feedback}

### 要求：
- 写作风格：{style}
- 预估字数：{word_count} 字左右
- 请完整撰写该章节内容，确保学术性和专业性
- 若有历史评审改进建议，请针对性地改进

请开始撰写："""),
        ])
        chain = prompt | self.llm | StrOutputParser()

        result = await chain.ainvoke({
            "section_title": section_title,
            "points": points_text,
            "references": refs_text,
            "feedback": feedback_text,
            "style": style,
            "word_count": word_count,
        })

        return {"content": result}

    async def stream_execute(self, input_data: Dict[str, Any]) -> AsyncIterator[str]:
        section_title = input_data.get("section_title", "")
        outline_points = input_data.get("outline_points", [])
        points_text = "\n".join(f"- {p}" for p in outline_points)

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", f"章节标题：{section_title}\n内容要点：\n{points_text}\n\n请开始撰写。"),
        ])
        chain = prompt | self.llm | StrOutputParser()

        async for chunk in chain.astream({}):
            yield chunk
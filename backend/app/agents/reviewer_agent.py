# reviewer_agent.py —— 学术评审专家 Agent（质量检查的 LLM 版）
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.agents.base_agent import BaseAgent
from app.utils.helpers import parse_json


class ReviewerAgent(BaseAgent):
    """以审稿人视角对论文做多维度打分，并给出可执行的改进建议"""

    def __init__(self):
        super().__init__(
            name="学术评审专家",
            description="对论文进行多维度质量评估，输出分数与改进建议",
        )

    def get_system_prompt(self) -> str:
        # JSON 示例中的花括号写成双份 {{ }}，避免被 ChatPromptTemplate 当成占位符
        return """你是一位严谨的学术论文评审专家，请以审稿人视角评审论文，从以下 5 个维度打分：
1. 创新性与学术价值
2. 逻辑结构与论证
3. 内容完整度与充分性
4. 语言表达与学术规范
5. 文献引用质量

打分规则：
- 每个维度给 0~1 之间的小数（保留两位）
- 综合分 score 取 5 个维度分数的平均值（保留两位）
- 严格只输出 JSON，不要任何解释或额外文字，格式如下：
{{
  "score": 0.82,
  "dimensions": [
    {{"dimension": "创新性与学术价值", "score": 0.8, "comment": "一句话点评"}},
    {{"dimension": "逻辑结构与论证", "score": 0.85, "comment": "一句话点评"}},
    {{"dimension": "内容完整度与充分性", "score": 0.8, "comment": "一句话点评"}},
    {{"dimension": "语言表达与学术规范", "score": 0.83, "comment": "一句话点评"}},
    {{"dimension": "文献引用质量", "score": 0.8, "comment": "一句话点评"}}
  ],
  "overall": "总体评价 + 最需要改进的地方（一段话，供作者修改参考）"
}}"""

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
{references}

请严格按系统要求的 JSON 格式输出评审结果："""),
        ])
        chain = prompt | self.llm | StrOutputParser()

        raw = await chain.ainvoke({
            "title": input_data.get("title", ""),
            "topic": input_data.get("topic", ""),
            "content": input_data.get("content", ""),
            "references": input_data.get("references", ""),
        })

        data = parse_json(raw) or {}
        return {"raw": raw, "data": data}
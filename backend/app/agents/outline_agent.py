# outline_agent.py —— 大纲生成 Agent（第 2 个）
# outline_agent.py —— 大纲专家 Agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.agents.base_agent import BaseAgent
from app.utils.helpers import parse_json

class OutlineAgent(BaseAgent):
    """大纲专家：根据论文标题生成结构化的论文大纲"""

    def __init__(self):
        super().__init__(
            name="大纲专家",
            description="根据论文标题生成结构化的论文大纲",
        )

    def get_system_prompt(self) -> str:
        # 注意：JSON 示例里的花括号全部写成双份 {{ }}，
        # 否则会被 ChatPromptTemplate 误当成占位符而报错。
        return """你是一位资深学术论文大纲专家，擅长根据标题设计清晰、有逻辑的论文结构。

要求：
1. 大纲需包含：摘要、引言、正文若干章、结论
2. 每章给出 2-4 个要点
3. 为每章合理分配字数
4. 严格按照下面的 JSON 格式输出，不要输出任何多余解释：
{{
  "title": "论文标题",
  "sections": [
    {{
      "title": "章节标题",
      "points": ["要点1", "要点2"],
      "word_count": 800
    }}
  ]
}}"""

    async def execute(self, inputs: dict) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", "请为下面的论文设计大纲：\n标题：{title}\n论文类型：{paper_type}\n总字数要求：约 {word_limit} 字"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        result = await chain.ainvoke(inputs)

        outline_dict = parse_json(result) or {}
        return {"outline": result, "outline_dict": outline_dict}

   
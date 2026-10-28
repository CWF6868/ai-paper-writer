# reviewer_agent.py —— 学术评审专家 Agent（质量检查的 LLM 版）
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.agents.base_agent import BaseAgent
from app.utils.helpers import parse_json
from app.utils.llm import get_llm


class ReviewerAgent(BaseAgent):
    """以审稿人视角对论文做多维度打分，并给出可执行的改进建议"""

    def __init__(self, llm: Optional[Any] = None):
        # 评审是「打分」任务，不是「创作」任务：必须用 temperature=0 保证同一篇文章
        # 得到可复现的分数。默认的 0.7 会让同一份内容在 0.36~0.54 之间随机漂移，
        # 导致「重写后分数反而更低」的错觉。
        super().__init__(
            name="学术评审专家",
            description="对论文进行多维度质量评估，输出分数与改进建议",
            llm=llm or get_llm(temperature=0.0),
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
- section_feedback 必须逐章给出：section 字段抄写正文里的章标题（去掉 ## 前缀，逐字一致），
  advice 只写「该章节自身」需要改什么。不要把全文的意见套到每一章上。
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
  "section_feedback": [
    {{"section": "引言", "advice": "补充研究背景中的具体数据来源，删去与摘要重复的表述"}},
    {{"section": "实验结果与分析", "advice": "补上消融实验的具体数值表格，说明各模块贡献"}}
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
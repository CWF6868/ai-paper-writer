# topic_agent.py —— 选题推荐 Agent（第 1 个）
from typing import Dict, Any, AsyncIterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.agents.base_agent import BaseAgent


class TopicAgent(BaseAgent):
    """根据研究领域/关键词，推荐 3-5 个创新性论文选题"""

    def __init__(self):
        super().__init__(
            name="选题专家",
            description="根据研究领域和关键词推荐创新性论文选题",
        )

    def get_system_prompt(self) -> str:
        # 重写人设，告诉大模型它的角色和输出要求
        return """你是一位资深的学术研究选题专家，擅长分析研究领域趋势并推荐创新性选题。

你的职责：
1. 分析用户输入的研究领域和关键词
2. 结合当前研究热点推荐 3-5 个创新性选题
3. 对每个选题说明研究价值、创新点和可行性

输出格式：
## 选题推荐

### 选题 1：[标题]
**研究方向**：[具体方向]
**研究价值**：[价值说明]
**创新点**：[创新点描述]
**可行性分析**：[可行性说明]

请确保推荐的选题具有学术价值、明确的创新点、合理的研究范围、可行的实施路径。"""

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 从入参中取研究领域和关键词
        field = input_data.get("field", "")
        keywords = input_data.get("keywords", "")
        requirements = input_data.get("requirements", "")

        # 拼一组「系统人设 + 用户提问」，交给大模型
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", """请根据以下信息推荐论文选题：

研究领域：{field}
关键词：{keywords}
{requirements}

请推荐 3-5 个创新性选题。"""),
        ])

        # 组装一条链：提示词 → 大模型 → 输出解析为纯文本
        chain = prompt | self.llm | StrOutputParser()

        result = await chain.ainvoke({
            "field": field,
            "keywords": keywords,
            "requirements": f"特殊要求：{requirements}" if requirements else "",
        })

        return {"topics": result}

    async def stream_execute(self, input_data: Dict[str, Any]) -> AsyncIterator[str]:
        # 流式版：边生成边吐出文字，前端可实时显示
        field = input_data.get("field", "")
        keywords = input_data.get("keywords", "")

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", f"研究领域：{field}\n关键词：{keywords}\n\n请推荐 3-5 个创新性选题。"),
        ])
        chain = prompt | self.llm | StrOutputParser()

        async for chunk in chain.astream({}):
            yield chunk
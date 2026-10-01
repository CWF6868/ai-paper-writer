# reference_agent.py —— 文献检索 Agent（第 4 个）
from typing import Dict, Any, AsyncIterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.agents.base_agent import BaseAgent
from app.rag.retriever import ReferenceRetriever


class ReferenceAgent(BaseAgent):
    """
    文献检索 Agent：
    1. 先用 RAG 按语义相似度检索相关文献
    2. 再让大模型把检索结果整理成标准引用格式
    """

    def __init__(self):
        super().__init__(
            name="文献专家",
            description="检索和推荐相关学术文献",
        )
        self.retriever = ReferenceRetriever()

    def get_system_prompt(self) -> str:
        return """你是一位资深的学术文献专家，擅长检索、分析和整理学术文献。

你的职责：
1. 根据研究主题推荐相关文献
2. 生成标准格式的引用文本（GB/T 7714、APA、MLA 等）
3. 分析文献的核心观点和参考价值

引用格式示例（GB/T 7714）：
[序号] 作者. 题名[J]. 刊名, 出版年, 卷(期): 起止页码.

请确保：引用信息完整准确、格式符合学术规范、文献与研究主题相关。"""

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        top_k = input_data.get("top_k", 5)
        format_style = input_data.get("format", "gbt")

        # 1) 用 RAG 从向量库检索相关文献
        retrieved_docs = await self.retriever.search(query, top_k=top_k)

        # 把检索到的文献元信息拼成文本，交给大模型生成引用
        docs_text = "\n\n".join(
            f"文献 {i+1}:\n"
            f"标题: {doc.get('title', 'N/A')}\n"
            f"作者: {doc.get('authors', 'N/A')}\n"
            f"期刊: {doc.get('journal', 'N/A')}\n"
            f"年份: {doc.get('year', 'N/A')}"
            for i, doc in enumerate(retrieved_docs)
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", """请将以下文献信息整理为标准引用格式：

检索主题：{query}
引用格式：{format_style}

文献信息：
{documents}

请输出编号的引用列表。"""),
        ])
        chain = prompt | self.llm | StrOutputParser()

        formatted_refs = await chain.ainvoke({
            "query": query,
            "format_style": format_style,
            "documents": docs_text,
        })

        return {
            "references": retrieved_docs,
            "formatted": formatted_refs,
        }

    async def stream_execute(self, input_data: Dict[str, Any]) -> AsyncIterator[str]:
        query = input_data.get("query", "")
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", f"请为选题「{query}」检索并整理相关文献引用列表。"),
        ])
        chain = prompt | self.llm | StrOutputParser()

        async for chunk in chain.astream({}):
            yield chunk
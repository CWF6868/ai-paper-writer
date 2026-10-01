# retriever.py —— 文献检索器（供 ReferenceAgent 调用）
from typing import List, Dict, Optional

from app.rag.vector_store import vector_store

class ReferenceRetriever:
    """
    文献检索器：封装 VectorStore 的检索能力。
    与文档原版的一个改进：把 metadata（标题/作者/期刊/年份）“拍平”到结果顶层，
    这样 ReferenceAgent 可以直接 doc.get('title') 取字段，不用再钻一层。
    """

    def __init__(self, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """按语义相似度检索文献，可附加过滤条件（如年份）"""
        k = top_k or self.top_k
        results = self.vector_store.search(query, top_k=k)

        docs = []
        for r in results:
            item = dict(r.get("metadata", {}))  # 元信息拍平到顶层
            item["id"] = r["id"]
            item["text"] = r["text"]
            item["score"] = r["score"]
            docs.append(item)

        # 过滤：只保留 metadata 中所有字段都匹配的文献
        if filters:
            docs = [
                d for d in docs
                if all(d.get(key) == value for key, value in filters.items())
            ]
        return docs

    async def search_by_topic(
        self,
        topic: str,
        year_range: Optional[tuple] = None,
    ) -> List[Dict]:
        """按主题检索，可限定年份范围，如 (2020, 2024)"""
        results = await self.search(topic)
        if year_range:
            start_year, end_year = year_range
            results = [
                d for d in results
                if d.get("year") and start_year <= int(d["year"]) <= end_year
            ]
        return results
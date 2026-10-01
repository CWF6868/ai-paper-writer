# vector_store.py —— ChromaDB 向量数据库管理（单例模式）
from typing import List, Dict
import os

import chromadb
from chromadb.config import Settings

from app.config import settings
from app.rag.embeddings import get_embeddings

class VectorStore:
    """
    向量数据库管理类（单例：全局只会有一个实例）。

    与文档原版的关键区别（修复坑一）：
    文档原版初始化了 Embedding 模型却从未使用，Chroma 实际用的是内置英文小模型。
    这里改为：先用百炼把文本算成向量，再把向量显式传给 Chroma 存储和查询，
    确保“入库”和“检索”两端的向量化都走百炼 text-embedding-v3。
    """

    _instance = None

    def __new__(cls):
        # 单例模式：已有实例就直接返回，不重复创建
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 确保存储目录存在（Chroma 不会自动创建多级目录）
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)

        # ChromaDB 持久化客户端：数据写入磁盘，程序重启不丢失
        self.client = chromadb.PersistentClient(
            path=settings.VECTOR_DB_PATH,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # 百炼向量化模型（修复坑一的关键）
        self.embeddings = get_embeddings()

        # 创建或获取集合（相当于数据库里的一张“表”）
        self.collection = self.client.get_or_create_collection(
            name="academic_references",
            metadata={"description": "学术论文参考文献向量库"},
        )

        self._initialized = True

    def add_documents(self, documents: List[Dict]) -> None:
        """
        把文献灌入向量库。
        每篇文献的格式：{"id": 编号, "text": 用于向量化的文本, "metadata": 标题/作者等元信息}
        用 upsert 而不是 add：id 已存在时覆盖更新，脚本重复跑也不会报错。
        """
        if not documents:
            return

        # 百炼 text-embedding-v3 单次请求最多 20 条，这里按 20 一批分组向量化并入库。
        # 不依赖 langchain 自带分批（它对 OpenAI 默认 chunk_size=1000，一次发超 20 条会 400）。
        batch_size = 20
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            ids = [doc["id"] for doc in batch]
            texts = [doc["text"] for doc in batch]
            metadatas = [doc.get("metadata", {}) for doc in batch]

            # 关键一步：调百炼把文本批量转成向量
            vectors = self.embeddings.embed_documents(texts)

            # 文本、元信息、向量一起存入 Chroma（upsert：id 存在则覆盖）
            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=vectors,
            )

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        相似度检索：把查询也用百炼向量化，再到 Chroma 里找距离最近的 top_k 篇文献。
        """
        count = self.collection.count()
        if count == 0:
            return []

        # 查询语句也要向量化（和入库用同一个模型，保证可比）
        query_vector = self.embeddings.embed_query(query)

        # n_results 不能超过库里的总数，否则 Chroma 会报错
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        for i in range(len(results["ids"][0])):
            documents.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                # 距离越小越相似，这里转成“越大越相似”的粗略分数
                "score": 1 - results["distances"][0][i],
            })
        return documents

    def delete_document(self, doc_id: str) -> None:
        """删除单篇文献"""
        self.collection.delete(ids=[doc_id])

    def get_document_count(self) -> int:
        """当前库里有多少篇文献"""
        return self.collection.count()

    def clear(self) -> None:
        """清空并重建集合"""
        self.client.delete_collection("academic_references")
        self.collection = self.client.get_or_create_collection(
            name="academic_references",
            metadata={"description": "学术论文参考文献向量库"},
        )

# 全局单例实例（别的文件 import 后直接可用）
vector_store = VectorStore()
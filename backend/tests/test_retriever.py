# 单元测试：rag/retriever.py（用假 VectorStore，不联网）
from app.rag.retriever import ReferenceRetriever


class FakeVectorStore:
    """假向量库：记录查询参数，返回预设原始结果（含 metadata 嵌套）"""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append({"query": query, "top_k": top_k})
        return list(self.results)


def _raw_results():
    return [
        {"id": "ref-010", "text": "RAG 摘要", "score": 0.83,
         "metadata": {"title": "RAG", "authors": "Lewis 等", "journal": "NeurIPS", "year": 2020}},
        {"id": "ref-002", "text": "BERT 摘要", "score": 0.42,
         "metadata": {"title": "BERT", "authors": "Devlin 等", "journal": "NAACL", "year": 2019}},
    ]


def _retriever(results=None, top_k=5):
    r = ReferenceRetriever(top_k=top_k)
    r.vector_store = FakeVectorStore(results if results is not None else _raw_results())
    return r


async def test_search_flattens_metadata_to_top_level():
    r = _retriever()
    docs = await r.search("RAG")
    assert docs[0]["title"] == "RAG"          # metadata 已拍平
    assert docs[0]["authors"] == "Lewis 等"
    assert docs[0]["id"] == "ref-010"
    assert docs[0]["text"] == "RAG 摘要"
    assert docs[0]["score"] == 0.83
    assert "metadata" not in docs[0]          # 不残留嵌套层


async def test_search_uses_default_top_k():
    r = _retriever(top_k=3)
    await r.search("任意")
    assert r.vector_store.calls == [{"query": "任意", "top_k": 3}]


async def test_search_top_k_argument_overrides_default():
    r = _retriever(top_k=3)
    await r.search("任意", top_k=1)
    assert r.vector_store.calls[0]["top_k"] == 1


async def test_search_applies_metadata_filters():
    r = _retriever()
    docs = await r.search("任意", filters={"year": 2019})
    assert [d["id"] for d in docs] == ["ref-002"]


async def test_search_filter_with_no_match_returns_empty():
    r = _retriever()
    assert await r.search("任意", filters={"journal": "不存在的期刊"}) == []


async def test_search_handles_empty_result_set():
    r = _retriever(results=[])
    assert await r.search("任意") == []


async def test_search_tolerates_missing_metadata_key():
    """Chroma 可能返回空 metadata，不应崩"""
    r = _retriever(results=[{"id": "x", "text": "t", "score": 0.1}])
    docs = await r.search("任意")
    assert docs[0]["id"] == "x"
    assert docs[0]["text"] == "t"


async def test_search_by_topic_filters_by_year_range():
    r = _retriever()
    docs = await r.search_by_topic("任意", year_range=(2019, 2019))
    assert [d["id"] for d in docs] == ["ref-002"]


async def test_search_by_topic_drops_docs_without_year():
    r = _retriever(results=[
        {"id": "no-year", "text": "t", "score": 0.5, "metadata": {"title": "X"}},
    ])
    assert await r.search_by_topic("任意", year_range=(2000, 2100)) == []


async def test_search_by_topic_without_range_returns_all():
    r = _retriever()
    docs = await r.search_by_topic("任意")
    assert len(docs) == 2

# 单元测试：rag/vector_store.py
#
# 隔离策略（很重要）：
#   1) 不使用 client.delete_collection()：删集合会触发 shutil.rmtree，
#      在本机沙箱里会撞上批量删除保护而中断整个测试会话；
#      因此改为「固定名测试集合 + 只清文档」。
#   2) 把单例的 collection 临时指向这个测试集合，并换上「假向量化模型」注入固定维度向量，
#      既不联网、不计费，也绝不碰生产集合 academic_references。
import pytest

from app.rag.vector_store import VectorStore, vector_store

TEST_COLLECTION = "unit_test_refs"


class FakeEmbeddings:
    """假向量化模型：按文本长度造一个 4 维向量，稳定可复现"""

    def embed_documents(self, texts):
        return [[float(len(t) % 5), 1.0, 0.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [1.0, 1.0, 0.0, 0.0]


class FakeEmbeddingsBroken(FakeEmbeddings):
    def embed_documents(self, texts):
        raise AssertionError("空输入不该触发向量化调用")

    def embed_query(self, text):
        raise AssertionError("空库检索不该触发向量化调用")


def _purge(collection):
    """只删文档、不删集合（避免触发目录级删除）"""
    ids = (collection.get() or {}).get("ids") or []
    if ids:
        collection.delete(ids=list(ids))


@pytest.fixture
def store(monkeypatch):
    col = vector_store.client.get_or_create_collection(name=TEST_COLLECTION)
    _purge(col)
    monkeypatch.setattr(vector_store, "collection", col)
    monkeypatch.setattr(vector_store, "embeddings", FakeEmbeddings())
    yield vector_store
    _purge(col)


def _docs(n):
    return [
        {"id": f"d-{i:03d}", "text": f"文献正文{i}",
         "metadata": {"title": f"论文{i}", "year": 2000 + i % 20}}
        for i in range(n)
    ]


# ---------- 单例 ----------

def test_vector_store_is_singleton():
    assert VectorStore() is vector_store
    assert VectorStore() is VectorStore()


# ---------- 空库行为 ----------

def test_search_on_empty_store_returns_empty_list(store, monkeypatch):
    monkeypatch.setattr(vector_store, "embeddings", FakeEmbeddingsBroken())
    assert store.get_document_count() == 0
    assert store.search("任意查询", top_k=5) == []


def test_add_empty_list_is_noop(monkeypatch):
    """空列表直接返回，连向量化都不该调用"""
    monkeypatch.setattr(vector_store, "embeddings", FakeEmbeddingsBroken())
    vector_store.add_documents([])          # 不抛异常即为通过


# ---------- 写入 / 检索 ----------

def test_add_documents_then_count(store):
    store.add_documents(_docs(3))
    assert store.get_document_count() == 3


def test_add_documents_batches_over_twenty(store):
    """单次请求上限 20 条：25 篇必须分批（2 批）写入，结果一条不少"""
    store.add_documents(_docs(25))
    assert store.get_document_count() == 25


def test_upsert_is_idempotent(store):
    """重复灌同一批数据不会翻倍（upsert 而非 add）"""
    docs = _docs(5)
    store.add_documents(docs)
    store.add_documents(docs)
    assert store.get_document_count() == 5


def test_search_returns_flattened_fields_and_score(store):
    store.add_documents(_docs(3))
    results = store.search("查一查", top_k=2)
    assert len(results) == 2
    first = results[0]
    assert set(first.keys()) >= {"id", "text", "metadata", "score"}
    assert isinstance(first["metadata"], dict)
    assert isinstance(first["score"], float)


def test_search_top_k_larger_than_collection_is_clamped(store):
    """n_results 不能超过库内总数，否则 Chroma 会报错"""
    store.add_documents(_docs(2))
    assert len(store.search("查", top_k=99)) == 2


# ---------- 删除文档 ----------

def test_delete_document_removes_one(store):
    store.add_documents(_docs(3))
    store.delete_document("d-000")
    assert store.get_document_count() == 2

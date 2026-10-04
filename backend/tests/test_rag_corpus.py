# 单元测试：rag/corpus.py（固定语料库的结构完整性）
#
# 语料是 RAG 的唯一数据源，结构一旦被改坏（id 重复 / 缺字段），
# 检索与引用编号都会出错，所以这些不变量必须由测试锁住。
from app.rag.corpus import ACADEMIC_REFERENCES

REQUIRED_META = {"title", "authors", "journal", "year"}


def test_corpus_size_is_42():
    assert len(ACADEMIC_REFERENCES) == 42


def test_ids_are_unique_and_well_formed():
    ids = [d["id"] for d in ACADEMIC_REFERENCES]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("ref-") and len(i) == 7 for i in ids)


def test_every_entry_has_non_empty_text_and_required_metadata():
    for doc in ACADEMIC_REFERENCES:
        assert doc["text"].strip(), doc["id"]
        meta = doc["metadata"]
        assert REQUIRED_META <= set(meta), doc["id"]
        assert str(meta["title"]).strip(), doc["id"]
        assert isinstance(meta["year"], int), doc["id"]
        assert 2000 <= meta["year"] <= 2030, doc["id"]


def test_known_classic_papers_are_present():
    """抽查几篇经典论文，防止语料被误删"""
    titles = {d["metadata"]["title"] for d in ACADEMIC_REFERENCES}
    for t in (
        "Attention Is All You Need",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "Deep Residual Learning for Image Recognition",
        "Generative Adversarial Networks",
    ):
        assert t in titles

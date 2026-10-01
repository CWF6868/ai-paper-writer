# seed_rag.py —— 灌入学术文献语料库 + 相似度检索自检
# 用法：
#   python seed_rag.py            # 幂等灌入（upsert，重复跑不会重复计费/报错）
#   python seed_rag.py --clear    # 先清空向量库再重新灌入（完全重建）
#   python seed_rag.py --check    # 只查不灌：看当前库内数量 + 检索自检
# 注意：灌库会真实调用百炼向量化 API（费用可忽略）。
import asyncio
import sys

from app.rag.vector_store import vector_store
from app.rag.retriever import ReferenceRetriever
from app.rag.corpus import ACADEMIC_REFERENCES

# 自检查询：覆盖不同主题，验证检索结果主题相关
SELF_CHECK_QUERIES = [
    "检索增强生成 RAG 缓解大模型幻觉",
    "大语言模型预训练与缩放定律",
    "图神经网络与知识图谱表示学习",
    "图像目标检测与图像分割",
]


async def _self_check():
    print("\n===== 检索自检 =====")
    retriever = ReferenceRetriever()
    for q in SELF_CHECK_QUERIES:
        results = await retriever.search(q, top_k=3)
        print(f"\n查询「{q}」命中：")
        if not results:
            print("   （无结果）")
            continue
        for doc in results:
            print(f"   - 《{doc.get('title')}》 {doc.get('authors')} {doc.get('year')}（相似度 {doc.get('score'):.3f}）")


async def main():
    clear = "--clear" in sys.argv
    check_only = "--check" in sys.argv

    if check_only:
        print(f"当前库内文献数：{vector_store.get_document_count()}")
        await _self_check()
        return

    if clear:
        vector_store.clear()
        print("已清空向量库")

    vector_store.add_documents(ACADEMIC_REFERENCES)
    print(f"入库完成，语料 {len(ACADEMIC_REFERENCES)} 篇，当前库内总数：{vector_store.get_document_count()}")

    await _self_check()


if __name__ == "__main__":
    asyncio.run(main())
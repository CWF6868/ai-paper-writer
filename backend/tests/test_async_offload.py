# 单元测试：阻塞调用线程池化（P0-2 第 2 步）
#
# 这一组测的不是「结果对不对」（那是 test_retriever / test_api_export 的事），
# 而是「调用发生在哪个线程、事件循环有没有被占住」——
# 这两点靠读代码看不出来，只能实测：
#   1. 检索/导出必须跑在非事件循环线程（否则 to_thread 白加）
#   2. 跑这些阻塞调用期间，事件循环仍能调度其他协程（用 tick 计数验证）
import asyncio
import threading
import time

from app.database import AsyncSessionLocal
from app.models.reference import Reference

CONTENT = "## 引言\n\n这是导出用的正文。\n\n- 要点一"


async def _paper(client, headers, with_reference: bool = False) -> int:
    r = await client.post("/api/papers/", json={"title": "导出测试论文", "topic": "NLP"}, headers=headers)
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    r = await client.put(f"/api/papers/{pid}", json={"content": CONTENT, "status": "completed"}, headers=headers)
    assert r.status_code == 200, r.text

    if with_reference:
        async with AsyncSessionLocal() as db:
            db.add(Reference(paper_id=pid, title="被引文献A", authors="张三", journal="测试刊", year=2020))
            await db.commit()
    return pid


async def _measure_ticks(coro):
    """跑 coro 期间数事件循环的 tick 次数。

    5ms 一跳：若被阻塞 200ms，最多只能 tick 0~1 次；正常让出则约 40 次。
    阈值取 10，宽到不怕 CI 抖动，又足以区分「阻塞」和「没阻塞」。
    """
    ticks = 0
    stop = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.005)

    task = asyncio.create_task(ticker())
    try:
        result = await coro
    finally:
        stop.set()
        await task
    return result, ticks


# ---------------- 向量检索 ----------------

async def test_vector_search_runs_in_worker_thread(monkeypatch):
    """检索必须离开事件循环线程：Chroma 查询 + 百炼 embedding 都是同步阻塞调用。"""
    from app.rag import retriever as retriever_mod

    seen = {}

    def fake_search(query, top_k=5):
        seen["thread"] = threading.get_ident()
        seen["args"] = (query, top_k)
        return [{"id": "r1", "text": "原文", "metadata": {"title": "文献A"}, "score": 0.88}]

    monkeypatch.setattr(retriever_mod.vector_store, "search", fake_search)

    docs = await retriever_mod.ReferenceRetriever().search("查询语句", top_k=3)

    assert seen["thread"] != threading.get_ident(), "检索仍在事件循环线程里执行"
    assert seen["args"] == ("查询语句", 3)      # 参数透传不能错位（query / top_k 顺序）
    assert docs[0]["title"] == "文献A"          # metadata 拍平逻辑不受影响
    assert docs[0]["score"] == 0.88


async def test_slow_vector_search_keeps_loop_responsive(monkeypatch):
    """检索耗时 200ms 时，事件循环必须还能跑别的协程。"""
    from app.rag import retriever as retriever_mod

    def slow_search(query, top_k=5):
        time.sleep(0.2)
        return []

    monkeypatch.setattr(retriever_mod.vector_store, "search", slow_search)

    _, ticks = await _measure_ticks(retriever_mod.ReferenceRetriever().search("q"))
    assert ticks >= 10, f"检索阻塞了事件循环（200ms 内只 tick 了 {ticks} 次）"


# ---------------- 导出 ----------------

async def test_export_docx_runs_in_worker_thread(client, headers, monkeypatch):
    """docx 生成必须在线程里跑，且在跑期间事件循环仍可调度。"""
    from app.api import export as export_api

    pid = await _paper(client, headers)
    seen = {}

    def slow_build(paper):
        seen["thread"] = threading.get_ident()
        seen["title"] = paper.title
        time.sleep(0.2)
        return b"PK\x03\x04fake-docx"

    monkeypatch.setattr(export_api, "build_docx", slow_build)

    resp, ticks = await _measure_ticks(
        client.get(f"/api/papers/{pid}/export", params={"format": "docx"}, headers=headers)
    )

    assert resp.status_code == 200
    assert resp.content == b"PK\x03\x04fake-docx"        # 线程里的返回值原样透出
    assert seen["thread"] != threading.get_ident(), "docx 仍在事件循环线程里生成"
    assert seen["title"] == "导出测试论文"                # ORM 对象跨线程可用
    assert ticks >= 10, f"导出阻塞了事件循环（200ms 内只 tick 了 {ticks} 次）"


async def test_export_pdf_runs_in_worker_thread(client, headers, monkeypatch):
    """PDF 同理（不依赖本机字体：这里替换掉 build_pdf 本身）。"""
    from app.api import export as export_api

    pid = await _paper(client, headers)
    seen = {}

    def slow_build(paper):
        seen["thread"] = threading.get_ident()
        time.sleep(0.2)
        return b"%PDF-1.4 fake"

    monkeypatch.setattr(export_api, "build_pdf", slow_build)

    resp, ticks = await _measure_ticks(
        client.get(f"/api/papers/{pid}/export", params={"format": "pdf"}, headers=headers)
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert seen["thread"] != threading.get_ident()
    assert ticks >= 10


async def test_export_with_loaded_references_crosses_thread_safely(client, headers):
    """带参考文献的真实导出：ORM 关系必须已预加载，线程里不能触发惰性加载。

    这是 to_thread 改造最容易踩的坑——把 ORM 对象丢进线程后，
    任何未加载的关系访问都会炸（跨线程拿不到 session）。
    """
    pid = await _paper(client, headers, with_reference=True)

    resp = await client.get(f"/api/papers/{pid}/export", params={"format": "docx"}, headers=headers)
    assert resp.status_code == 200

    from docx import Document
    from io import BytesIO

    text = "\n".join(p.text for p in Document(BytesIO(resp.content)).paragraphs)
    assert "被引文献A" in text          # 文献真的写进了文档，不只是"没报错"

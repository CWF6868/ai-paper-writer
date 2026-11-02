# 单元测试：章节写作受控并发（P0-2）
#
# 这一组锁的是「并发改造不能顺手破坏的三件事」：
#   1. 并发确实发生（否则改造等于没做，性能故事是假的）
#   2. 并发上限确实生效（否则一次把 8 个请求砸给 DeepSeek，换来 429）
#   3. 章节顺序确实是大纲顺序（并发完成顺序不确定，写错就是论文乱序）
# 外加一条：单章失败必须能定位到章，而不是抛个裸异常让人猜。
#
# 注意：PaperWorkflowGraph() 构造一次约 5s（Chroma 客户端初始化），
# 所以用模块级 fixture 共享实例；每个用例自己换掉 writer_agent.execute。
import asyncio
import time

import pytest

from app.config import settings
from app.workflows.graph_builder import PaperWorkflowGraph


@pytest.fixture(scope="module")
def graph():
    return PaperWorkflowGraph()


def _outline(n: int) -> dict:
    return {"sections": [
        {"title": f"第{i}章 测试章节", "points": ["要点"], "word_count": 100}
        for i in range(1, n + 1)
    ]}


def _state(outline: dict, **over) -> dict:
    state = {
        "paper_id": 1, "title": "标题", "topic": "选题",
        "outline": outline, "references": [], "sections_content": {},
    }
    state.update(over)
    return state


class _SpyWriter:
    """记录调用顺序、在途峰值；sleep 时长可按章节标题定制（用来制造乱序完成）。"""

    def __init__(self, delays=None):
        self.delays = delays or {}
        self.calls = []
        self.in_flight = 0
        self.peak = 0

    async def __call__(self, payload: dict) -> dict:
        self.calls.append(payload)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delays.get(payload["section_title"], 0))
            return {"content": f"《{payload['section_title']}》正文"}
        finally:
            self.in_flight -= 1


async def test_write_sections_actually_runs_in_parallel(graph, monkeypatch):
    """8 章 × 0.15s：串行 1.2s，并发 4 应压到 ~0.3s（2 批）。

    断言用「明显小于串行」的宽松阈值，避免 CI 抖动导致假失败；
    真正锁死并发事实的是 peak == 4（在途请求峰值）。
    """
    monkeypatch.setattr(settings, "SECTION_CONCURRENCY", 4, raising=False)
    spy = _SpyWriter({f"第{i}章 测试章节": 0.15 for i in range(1, 9)})
    graph.writer_agent.execute = spy

    t0 = time.perf_counter()
    out = await graph._write_sections(_state(_outline(8)))
    elapsed = time.perf_counter() - t0

    assert spy.peak == 4, f"并发数应为 4，实际 {spy.peak}"
    assert elapsed < 8 * 0.15 * 0.7, f"未见并发收益：{elapsed:.2f}s（串行约 1.2s）"
    assert len(out["sections_content"]) == 8


async def test_write_sections_respects_semaphore_limit(graph, monkeypatch):
    """并发上限生效：调成 2 就最多 2 个在途，不能偷偷全并发。"""
    monkeypatch.setattr(settings, "SECTION_CONCURRENCY", 2, raising=False)
    spy = _SpyWriter({f"第{i}章 测试章节": 0.05 for i in range(1, 7)})
    graph.writer_agent.execute = spy

    await graph._write_sections(_state(_outline(6)))
    assert spy.peak == 2


async def test_write_sections_never_deadlocks_on_zero_concurrency(graph, monkeypatch):
    """配置成 0 / 负数时不能死锁：必须夹到 1，最差退化成串行。"""
    monkeypatch.setattr(settings, "SECTION_CONCURRENCY", 0, raising=False)
    spy = _SpyWriter()
    graph.writer_agent.execute = spy

    out = await asyncio.wait_for(graph._write_sections(_state(_outline(3))), timeout=5)
    assert spy.peak == 1
    assert len(out["sections_content"]) == 3


async def test_write_sections_preserves_outline_order(graph, monkeypatch):
    """完成顺序打乱（第1章最慢、第8章最快）时，成文顺序仍必须是大纲顺序。

    sections_content 的插入序 = 成文序（_polish_paper 拼接、_save_results 生成
    order_index 都依赖它）。这里若按完成顺序写入，论文会从「绪论」开始倒着排。
    """
    monkeypatch.setattr(settings, "SECTION_CONCURRENCY", 8, raising=False)
    # 越靠前的章越慢 → 完成顺序与大纲顺序完全相反
    spy = _SpyWriter({f"第{i}章 测试章节": (9 - i) * 0.03 for i in range(1, 9)})
    graph.writer_agent.execute = spy

    out = await graph._write_sections(_state(_outline(8)))

    assert list(out["sections_content"]) == [s["title"] for s in _outline(8)["sections"]]
    assert out["sections_content"]["第1章 测试章节"] == "《第1章 测试章节》正文"


async def test_write_sections_failure_names_the_section(graph, monkeypatch):
    """单章失败 → 报错里必须带章节名和原始异常，不能只抛一个裸 RuntimeError。"""
    monkeypatch.setattr(settings, "SECTION_CONCURRENCY", 4, raising=False)

    async def flaky(payload):
        if payload["section_title"].startswith("第2章"):
            raise ValueError("rate limited")
        await asyncio.sleep(0.01)
        return {"content": "正常正文"}

    graph.writer_agent.execute = flaky

    with pytest.raises(RuntimeError) as ei:
        await graph._write_sections(_state(_outline(4)))
    msg = str(ei.value)
    assert "第2章" in msg and "rate limited" in msg
    assert "1/4" in msg                       # 失败章数/总章数
    assert isinstance(ei.value.__cause__, ValueError)   # 原始异常链没丢


async def test_write_sections_rewrite_keeps_per_section_advice(graph, monkeypatch):
    """重写轮并发下，每章拿到的仍是自己的分章建议（不能串到别章去）。

    章节名用真实大纲那种互不相同的名字：归一化会剥掉「第X章」前缀，
    若各章去掉前缀后同名，_pick_section_advice 本来就会命中第一条
    （既有行为，与并发无关，也在 _norm_section_name 的兜底范围之外）。
    """
    monkeypatch.setattr(settings, "SECTION_CONCURRENCY", 4, raising=False)
    titles = ["绪论", "相关工作", "研究方法"]
    outline = {"sections": [
        {"title": t, "points": ["要点"], "word_count": 100} for t in titles
    ]}
    spy = _SpyWriter({t: 0.02 for t in titles})
    graph.writer_agent.execute = spy

    await graph._write_sections(_state(
        outline,
        quality_comment="整篇意见",
        quality_section_feedback=[
            {"section": "第1章 绪论", "advice": "补背景", "severity": "major"},
            {"section": "3. 研究方法", "advice": "补实验", "severity": "blocking"},
        ],
    ))

    got = {c["section_title"]: c["feedback"] for c in spy.calls}
    assert got["绪论"] == "补背景"
    assert got["研究方法"] == "补实验"
    assert got["相关工作"] == "整篇意见"     # 无专属建议 → 退回整篇意见


async def test_write_sections_keeps_previous_round_content_on_untouched_keys(graph, monkeypatch):
    """重写轮不得丢掉 state 里已有的章节内容（并发回填只覆盖本轮写的章）。"""
    monkeypatch.setattr(settings, "SECTION_CONCURRENCY", 2, raising=False)
    graph.writer_agent.execute = _SpyWriter()

    out = await graph._write_sections(_state(
        _outline(2),
        sections_content={"附录": "上一轮内容", "第1章 测试章节": "旧内容"},
    ))
    assert out["sections_content"]["附录"] == "上一轮内容"
    assert out["sections_content"]["第2章 测试章节"] == "《第2章 测试章节》正文"


def test_llm_client_carries_retry_budget():
    """并发抬高后 429 概率上升：LLM 客户端必须真的带上重试次数。"""
    from app.utils.llm import get_llm

    llm = get_llm()
    assert llm.max_retries == settings.LLM_MAX_RETRIES
    assert settings.LLM_MAX_RETRIES >= 1

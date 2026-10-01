# 单元测试：评审 Agent 的 JSON 解析（用假 LLM，不联网）
from langchain_core.language_models.fake import FakeListLLM

from app.agents.reviewer_agent import ReviewerAgent

VALID_JSON = """```json
{
  "score": 0.82,
  "dimensions": [
    {"dimension": "创新性与学术价值", "score": 0.8, "comment": "有一定创新"},
    {"dimension": "逻辑结构与论证", "score": 0.85, "comment": "论证清晰"},
    {"dimension": "内容完整度与充分性", "score": 0.8, "comment": "内容完整"},
    {"dimension": "语言表达与学术规范", "score": 0.83, "comment": "语言规范"},
    {"dimension": "文献引用质量", "score": 0.8, "comment": "引用恰当"}
  ],
  "overall": "总体不错，建议补充对比实验。"
}
```"""


def _agent_with(response: str) -> ReviewerAgent:
    agent = ReviewerAgent()
    agent.llm = FakeListLLM(responses=[response])
    return agent


async def test_reviewer_parses_valid_json():
    agent = _agent_with(VALID_JSON)
    result = await agent.execute({
        "title": "测试论文", "topic": "NLP", "content": "正文", "references": "文献",
    })
    data = result["data"]
    assert data["score"] == 0.82
    assert len(data["dimensions"]) == 5
    assert data["dimensions"][0]["dimension"] == "创新性与学术价值"
    assert "对比实验" in data["overall"]


async def test_reviewer_handles_invalid_json():
    """非法 JSON 不崩溃，data 里带原始文本"""
    agent = _agent_with("这不是 JSON 输出")
    result = await agent.execute({
        "title": "T", "topic": "NLP", "content": "C", "references": "R",
    })
    assert "raw" in result["data"]


async def test_reviewer_handles_empty_response():
    agent = _agent_with("")
    result = await agent.execute({
        "title": "T", "topic": "NLP", "content": "C", "references": "R",
    })
    assert result["data"] == {}

# 单元测试：agents/base_agent.py（公共模板）
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.base_agent import BaseAgent


class _EchoAgent(BaseAgent):
    """最小可用的具体 Agent：把入参原样吐回"""

    def __init__(self, **kw):
        super().__init__(name="回显专家", description="把输入原样返回", **kw)

    async def execute(self, input_data):
        return {"output": str(input_data)}


def test_default_system_prompt_mentions_role():
    a = _EchoAgent()
    prompt = a.get_system_prompt()
    assert "回显专家" in prompt and "把输入原样返回" in prompt


def test_add_message_supports_all_roles():
    a = _EchoAgent()
    a.add_message("system", "s")
    a.add_message("user", "u")
    a.add_message("assistant", "a")
    assert isinstance(a.memory[0], SystemMessage)
    assert isinstance(a.memory[1], HumanMessage)
    assert isinstance(a.memory[2], AIMessage)


def test_add_message_unknown_role_falls_back_to_human():
    a = _EchoAgent()
    a.add_message("tool", "t")
    assert isinstance(a.memory[0], HumanMessage)


def test_clear_memory_empties_history():
    a = _EchoAgent()
    a.add_message("user", "hi")
    assert a.memory
    a.clear_memory()
    assert a.memory == []


def test_repr_shows_class_and_name():
    assert repr(_EchoAgent()) == "<_EchoAgent(name='回显专家')>"


async def test_stream_execute_defaults_to_single_chunk():
    a = _EchoAgent()
    chunks = [c async for c in a.stream_execute({"x": 1})]
    assert chunks == ["{'x': 1}"]


def test_base_agent_is_abstract():
    """抽象基类不能被直接实例化（execute 未实现）"""
    with pytest.raises(TypeError):
        BaseAgent(name="x", description="y")

# base_agent.py —— Agent 基类（所有 Agent 的公共模板）
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, AsyncIterator

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.utils.llm import get_llm


class BaseAgent(ABC):
    """
    所有 Agent 都继承这个类。它统一管理：
      - name/description：这个 Agent 是谁、干什么
      - llm：它的大脑（DeepSeek）
      - memory：简单对话记忆
    子类只需实现 execute()（真正干活的方法）。
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm: Optional[Any] = None,
    ):
        self.name = name
        self.description = description
        self.llm = llm or get_llm()  # 没传就用默认 DeepSeek
        self.memory: list = []       # 对话记忆

    def add_message(self, role: str, content: str) -> None:
        """把一条消息加入记忆。role 支持 system/user/assistant。"""
        message_map = {
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage,
        }
        cls = message_map.get(role)
        if cls is None:
            cls = HumanMessage
        self.memory.append(cls(content=content))

    def clear_memory(self) -> None:
        """清空对话记忆"""
        self.memory = []

    def get_system_prompt(self) -> str:
        """给子类的默认人设；子类可重写为更具体的人设"""
        return f"你是{self.name}，{self.description}。请根据用户需求完成任务。"

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务，子类必须实现"""
        raise NotImplementedError

    async def stream_execute(self, input_data: Dict[str, Any]) -> AsyncIterator[str]:
        """流式输出（可选实现）；默认先整段执行再一次性输出"""
        result = await self.execute(input_data)
        yield result.get("output", "")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"
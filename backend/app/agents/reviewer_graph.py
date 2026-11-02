# reviewer_graph.py —— 评审 Agent 的显式子图（agent → tools → finalize）
#
# 不用 create_react_agent 的原因：黑盒没法加预算控制，而预算是必需的。
#
#   START → agent ──(有 tool_calls 且未超预算)──→ tools → agent ...
#          └──(无 tool_calls 或超预算)──→ finalize → END
#
# finalize 节点是必须的：模型调完工具后可能输出自然语言而不是 JSON，
# 这一节点做最后一次结构化收口，保证出口类型恒定为 ReviewResult。
from typing import Annotated, Any, List, Optional, Sequence, Tuple

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from app.llm.schemas import ReviewResult
from app.llm.structured import StructuredOutputError, call_structured, log_llm_call
from app.llm.tools.registry import MAX_TOOL_CALLS, MAX_TOOL_ROUNDS
from app.utils.helpers import parse_json

import time


class ReviewerState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    tool_rounds: int          # 已完成的「模型→工具」往返轮数
    result: ReviewResult      # finalize 的结构化产出
    mode: str                 # finalize 实际成功的结构化输出模式


def _to_message(resp: Any) -> BaseMessage:
    """把模型返回值统一成 BaseMessage（FakeListLLM 直接返回 str，也兼容）。"""
    if isinstance(resp, str):
        return AIMessage(content=resp)
    if isinstance(resp, BaseMessage):
        return resp
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        return AIMessage(content="".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        ))
    return AIMessage(content=str(content))


def _bind_tools(model: Any, tools: Sequence) -> Any:
    """防御性绑定：模型不支持 tools（Fake 模型/旧客户端）就裸调。"""
    if not tools:
        return model
    try:
        return model.bind_tools(tools)
    except Exception:
        return model


def _count_tool_calls(messages: Sequence[AnyMessage]) -> int:
    """累计工具调用次数（预算 MAX_TOOL_CALLS 的口径）。"""
    return sum(
        len(getattr(m, "tool_calls", None) or [])
        for m in messages
    )


def _last_ai_content(messages: Sequence[AnyMessage]) -> Optional[str]:
    """从后往前找最后一条 AI 文本输出（finalize 快路径用）。"""
    for m in reversed(list(messages)):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content
    return None


def clean_messages_for_finalize(
    messages: Sequence[AnyMessage],
    system_prompt: Optional[str] = None,
) -> List[AnyMessage]:
    """收口前的消息清洗：DeepSeek 要求 tool_calls 必须成对、tool 消息必须有前置 tool_calls。

    预算强制收口时，末条 AIMessage 带着「未配对的 tool_calls」，直接发会 400；
    剥掉 tool_calls 又会让历史里的 tool 结果失去配对对象。因此：
    把所有带 tool_calls 的 AIMessage 与紧跟其后的 tool 消息一并裁掉，
    只保留干净的 system/human + 纯文本 AI 上下文，再追加一句收口指令。

    system_prompt 非空时，用它替换原 system 消息（剔除工具说明，
    防止模型在收口时幻调 verify_citation 导致结构化解析 KeyError）。
    """
    out: List[AnyMessage] = []
    skip_tool_msgs = False
    system_replaced = False
    for m in messages:
        if system_prompt and not system_replaced and getattr(m, "type", None) == "system":
            out.append(SystemMessage(content=system_prompt))
            system_replaced = True
            continue
        is_ai = isinstance(m, AIMessage)
        has_calls = bool(getattr(m, "tool_calls", None))
        if is_ai and has_calls:
            # 裁掉这条 tool_calls 消息，并标记后续 tool 结果也裁掉
            skip_tool_msgs = True
            continue
        if skip_tool_msgs:
            if getattr(m, "type", None) == "tool":
                continue
            skip_tool_msgs = False
        out.append(m)
    return out


def build_reviewer_graph(
    model: Any,
    tools: Optional[Sequence] = None,
    finalize_system_prompt: Optional[str] = None,
):
    """构建评审子图。tools 显式传参：单测可注入假工具，传 [] 则禁用工具环。

    finalize_system_prompt：收口时用来替换原 system prompt（不含工具说明）。
    收口只绑定 ReviewResult，若 system 仍提及 verify_citation，模型会幻觉调用它，
    结构化解析器找不到对应 schema → KeyError → function_calling 级失败。
    """
    from app.llm.tools.registry import REVIEWER_TOOLS
    tool_list = list(tools) if tools is not None else list(REVIEWER_TOOLS)
    tool_node = ToolNode(tool_list) if tool_list else None

    async def call_model(state: ReviewerState) -> dict:
        bound = _bind_tools(model, tool_list)
        t0 = time.perf_counter()
        resp = await bound.ainvoke(state["messages"])
        log_llm_call("reviewer_agent", "chat", "ok", (time.perf_counter() - t0) * 1000)
        return {"messages": [_to_message(resp)]}

    def route(state: ReviewerState) -> str:
        if tool_node is None:
            return "finalize"          # 工具环被显式禁用
        last = state["messages"][-1]
        pending = getattr(last, "tool_calls", None) or []
        if not pending:
            return "finalize"
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            return "finalize"          # 轮数超限：强制收口，不再进工具
        if _count_tool_calls(state["messages"]) >= MAX_TOOL_CALLS:
            return "finalize"          # 累计调用数超限
        return "tools"

    async def run_tools(state: ReviewerState) -> dict:
        out = await tool_node.ainvoke(state)
        out["tool_rounds"] = state.get("tool_rounds", 0) + 1
        return out

    async def finalize(state: ReviewerState) -> dict:
        # 快路径：最后的 AI 输出已经能直接解析成 ReviewResult → 不再花一次 LLM 调用
        content = _last_ai_content(state["messages"])
        if content:
            parsed_json = parse_json(content)
            if not (set(parsed_json.keys()) == {"raw"} or not parsed_json):
                try:
                    return {"result": ReviewResult.model_validate(parsed_json), "mode": "raw_parse"}
                except Exception:
                    pass  # 内容不是合法评审结构 → 走结构化收口
        # 收口：裁掉未完成的工具往返后，做最后一次 with_structured_output，
        # 保证出口类型恒定为 ReviewResult。追加 HumanMessage（不能以 AIMessage 结尾）。
        clean = clean_messages_for_finalize(state["messages"], finalize_system_prompt)
        clean.append(HumanMessage(content="请基于上述评审材料，直接输出符合要求的评审结果。"))
        parsed, mode = await call_structured(model, ReviewResult, clean,
                                             node="reviewer_finalize")
        return {"result": parsed, "mode": mode}

    workflow = StateGraph(ReviewerState)
    workflow.add_node("agent", call_model)
    if tool_node is not None:
        workflow.add_node("tools", run_tools)
    else:
        workflow.add_node("tools", _tools_disabled)
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route, {"tools": "tools", "finalize": "finalize"})
    workflow.add_edge("tools", "agent")
    workflow.add_edge("finalize", END)
    return workflow.compile()


async def _tools_disabled(state: ReviewerState) -> dict:
    """tools 传 [] 时不可达（route 永远走 finalize），占位保证图结构完整。"""
    return {"tool_rounds": state.get("tool_rounds", 0)}


async def run_reviewer(
    model: Any,
    messages: Sequence[Any],
    tools: Optional[Sequence] = None,
    finalize_system_prompt: Optional[str] = None,
) -> Tuple[ReviewResult, str]:
    """跑一次评审子图，返回 (ReviewResult, mode)。三级全失败时抛 StructuredOutputError。"""
    graph = build_reviewer_graph(model, tools, finalize_system_prompt)
    state = await graph.ainvoke({"messages": list(messages)})
    if "result" not in state or state["result"] is None:
        raise StructuredOutputError("reviewer_graph 未能产出结构化评审结果")
    return state["result"], state["mode"]

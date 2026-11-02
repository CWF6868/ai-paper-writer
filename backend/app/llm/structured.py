# structured.py —— with_structured_output 封装：三级降级 + 观测日志
#
# 三级降级：function_calling → json_mode → 裸文本解析。
# 关键原则：每一级都记录，不静默。全失败抛 StructuredOutputError，
# 由调用方显式降级（quality_degraded），绝不编造数据。
#
# 观测：结构化 JSON 日志（logger 名 app.llm.observability），
# 字段与设计文档 4.2 节 llm_call_log 对齐；表结构留到 P0-3 成本归因一起做。
import json
import logging
import time
from typing import Any, List, Optional, Sequence, Tuple, Type, TypeVar

from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.utils.helpers import parse_json

logger = logging.getLogger("app.llm.observability")

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """三级降级全部失败。调用方必须显式处理（降级标记），不允许静默编数据。"""


def log_llm_call(node: str, mode: str, status: str, latency_ms: float, error: Optional[str] = None) -> None:
    """结构化观测日志：一行 JSON，后续接入 llm_call_log 表时字段直接对齐。"""
    logger.info(
        "LLM_CALL %s",
        json.dumps({
            "node": node,
            "mode": mode,
            "status": status,           # ok / fail
            "latency_ms": round(latency_ms, 1),
            "error": (error or "")[:300],
        }, ensure_ascii=False),
    )


def log_tool_call(node: str, tool_name: str, ok: bool, latency_ms: float, args_digest: str = "") -> None:
    """工具调用观测日志（agent_tool_log 口径）。args 只记摘要指纹，不记原文（控日志体积）。"""
    logger.info(
        "TOOL_CALL %s",
        json.dumps({
            "node": node,
            "tool": tool_name,
            "ok": ok,
            "latency_ms": round(latency_ms, 1),
            "args_digest": args_digest,
        }, ensure_ascii=False),
    )


def _message_text(resp: Any) -> str:
    """把模型返回值统一成文本：兼容 AIMessage / str / RunnableLambda 输出。"""
    if isinstance(resp, str):
        return resp
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        # content blocks：[{type: text, text: ...}, ...]
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


async def _raw_parse(llm: Any, schema: Type[T], messages: Sequence[Any]) -> T:
    """第三级：裸文本解析。剥 Markdown 代码块后 json.loads + Pydantic 校验。"""
    resp = await llm.ainvoke(messages)
    text = _message_text(resp)
    parsed_json = parse_json(text)
    # parse_json 失败时返回 {"raw": 原文} —— 单键 raw 即解析失败信号
    if set(parsed_json.keys()) == {"raw"} or not parsed_json:
        raise ValueError(f"裸文本不是合法 JSON: {text[:120]!r}")
    return schema.model_validate(parsed_json)


def _ensure_json_hint(messages: Sequence[Any]) -> Sequence[Any]:
    """DeepSeek 的 response_format={"type":"json_object"} 硬性要求 prompt 里出现 'json'。

    实测（2026-11-01）：收口指令「请直接输出评审结果」不含 json 字样，
    json_mode 这一级必然 400 失败 —— 等于第二级降级长期失效。
    这里在发请求前检查，缺了就补一条指令（不污染调用方原始消息）。
    """
    texts = []
    for m in messages:
        content = getattr(m, "content", m)
        texts.append(content if isinstance(content, str) else str(content))
    if "json" in " ".join(texts).lower():
        return messages
    return list(messages) + [
        HumanMessage(content="请仅以 JSON 格式输出结果，不要输出 JSON 之外的任何解释文字。")
    ]


async def call_structured(
    llm: Any,
    schema: Type[T],
    messages: Sequence[Any],
    *,
    node: str,
) -> Tuple[T, str]:
    """三级降级调用：返回 (schema 实例, 实际成功的模式名)。

    mode ∈ {"function_calling", "json_mode", "raw_parse"}，
    调用方以 mode != "function_calling" 判定 degraded。
    """
    attempts: List[Tuple[str, Optional[dict]]] = [
        ("function_calling", {"method": "function_calling"}),
        ("json_mode", {"method": "json_mode"}),
        ("raw_parse", None),
    ]
    last_error: Optional[Exception] = None
    for mode, kw in attempts:
        t0 = time.perf_counter()
        # json_mode 必须先满足「prompt 含 json」这个前置条件，否则必然 400
        attempt_messages = _ensure_json_hint(messages) if mode == "json_mode" else messages
        try:
            if kw is None:
                parsed = await _raw_parse(llm, schema, attempt_messages)
            else:
                if not hasattr(llm, "with_structured_output"):
                    raise AttributeError(f"{type(llm).__name__} 不支持 with_structured_output")
                structured = llm.with_structured_output(schema, include_raw=True, **kw)
                out = await structured.ainvoke(attempt_messages)
                if out.get("parsing_error"):
                    raise ValueError(str(out["parsing_error"]))
                parsed = out.get("parsed")
                if parsed is None:
                    raise ValueError("parsed 为 None")
            log_llm_call(node, mode, "ok", (time.perf_counter() - t0) * 1000)
            return parsed, mode
        except Exception as e:  # noqa: BLE001 —— 每一级的失败都吞掉并记录，进入下一级
            last_error = e
            log_llm_call(node, mode, "fail", (time.perf_counter() - t0) * 1000, error=str(e))
            continue
    raise StructuredOutputError(f"{node} 三级降级全部失败，最后错误：{last_error}")

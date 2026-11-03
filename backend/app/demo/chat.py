# demo/chat.py —— 演示模式的「假模型」：实现真实的 LangChain 聊天模型接口
#
# 为什么不是简单地在 API 层返回假 JSON：
# 那样只能演示前端，会绕过 P0-1 的三级降级、P0-2 的并发、P0-3 的计量。
# 这里做一个 BaseChatModel 子类，让 prompt|llm|parser、bind_tools、
# with_structured_output（function_calling / json_mode / raw_parse 三级）全部照常工作，
# 上层一行代码都不用改 —— 只是在 build_chat_model 的分支里换了实现。
import json
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda

from app.config import settings
from app.demo.content import (
    build_citation_verdict,
    build_generic,
    build_outline,
    build_reference_list,
    build_review,
    build_section,
    build_topic_list,
)

# 节点识别靠 system prompt 的人设关键词。
# 用关键词而不是"第几个调用"：流水线会重写、会并发，顺序不可靠，人设是稳定的。
_AGENT_MARKERS: List[Tuple[str, str]] = [
    ("学术论文大纲专家", "outline"),
    ("学术论文评审专家", "reviewer"),
    ("学术论文写作专家", "writer"),
    ("学术论文润色专家", "polish"),
    ("学术研究选题专家", "topic"),
    ("学术文献专家", "reference"),
]


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


def _system_text(messages: Sequence[BaseMessage]) -> str:
    for m in messages:
        if getattr(m, "type", None) == "system":
            return _message_text(m)
    return ""


def _last_human_text(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(list(messages)):
        if getattr(m, "type", None) in ("human", "user"):
            return _message_text(m)
    return ""


def _richest_human_text(messages: Sequence[BaseMessage], *needles: str) -> str:
    """取「信息量最大的那条人类消息」。

    为什么不能只取最后一条：评审子图收口时会追加一句
    「请基于上述评审材料，直接输出评审结果」，最后一条人类消息就变成了这句指令，
    正文和参考文献都在它前面。按最后一条解析，会把评审对象解析成一句指令
    （实测后果：分章建议为空、引用维度被判成"没有引用"、总分被压低）。
    """
    candidates = [
        _message_text(m) for m in messages
        if getattr(m, "type", None) in ("human", "user")
    ]
    for needle in needles:
        for text in candidates:
            if needle in text:
                return text
    return max(candidates, key=len) if candidates else ""


def detect_node(messages: Sequence[BaseMessage]) -> str:
    """根据 system prompt 判定当前是哪个 Agent 在调用。"""
    system = _system_text(messages)
    for marker, node in _AGENT_MARKERS:
        if marker in system:
            return node
    return "generic"


def _extract_polish_source(prompt: str) -> str:
    """润色 Prompt 里正文夹在「## 原文内容：」与「## 输出要求：」之间。"""
    start = prompt.find("## 原文内容：")
    end = prompt.find("## 输出要求：")
    if start == -1:
        return prompt
    body = prompt[start + len("## 原文内容："): end if end != -1 else len(prompt)]
    return body.strip()


def reply_text(messages: Sequence[BaseMessage]) -> str:
    """按节点生成文本。任何异常都必须有兜底，演示环境不能因为一次解析失败就 500。"""
    node = detect_node(messages)
    prompt = _last_human_text(messages)
    if node == "writer":
        return build_section(prompt)
    if node == "polish":
        # 润色：原样返回。演示模式不伪造改写效果，也保证不会把正文写丢
        # （生产代码里 <40% 保留率会回退原文，这里直接就是原文，行为一致）。
        return _extract_polish_source(prompt)
    if node == "topic":
        return build_topic_list(prompt)
    if node == "reference":
        return build_reference_list(prompt)
    if node == "reviewer":
        # 评审 agent 轮：故意返回自然语言（不是 JSON），
        # 让 finalize 走真实的结构化收口路径，而不是命中 raw_parse 快路径。
        return (
            "已通读全文。结构与论证基本完整，创新点表述偏弱，"
            "中段缺少支撑材料，引用密度在章节间不均衡。下面给出结构化评审结果。"
        )
    if node == "outline":
        # 大纲走 with_structured_output；这里只在 raw_parse 兜底时命中
        return json.dumps(build_outline(prompt), ensure_ascii=False)
    return build_generic(prompt)


_STREAM_BREAKS = "。！？；，\n"


def iter_stream_chunks(text: str, size: int = 26) -> Iterator[str]:
    """把整段文本切成小片，让 SSE 真的逐段推进（而不是一次性刷完）。

    按标点断片，避免把「检索增强」这类词切成两半；
    连续无标点的长串超过 2 倍片长时强制切断，防止出现超大分片。
    """
    buf = ""
    for ch in text or "":
        buf += ch
        if len(buf) >= size and ch in _STREAM_BREAKS:
            yield buf
            buf = ""
        elif len(buf) >= size * 2:
            yield buf
            buf = ""
    if buf:
        yield buf


def _schema_payload(schema: Any, messages: Sequence[BaseMessage]) -> Dict[str, Any]:
    """结构化输出：按 Schema 类型生成对应载荷（仍然只换模型，不换校验）。"""
    name = getattr(schema, "__name__", "")
    if name == "OutlineResult":
        prompt = _richest_human_text(messages, "设计大纲", "标题：")
        return build_outline(prompt)
    if name == "ReviewResult":
        prompt = _richest_human_text(messages, "## 正文内容", "## 论文标题")
        return build_review(prompt)
    if name == "CitationVerdict":
        prompt = _richest_human_text(messages, "相似度")
        return build_citation_verdict(prompt)
    raise ValueError(f"演示模式未覆盖的结构化 Schema: {name}")


def _estimate_usage(messages: Sequence[BaseMessage], output: str) -> Dict[str, int]:
    """粗略 token 估算：只为让成本表有非零的 token 数，成本仍按 ¥0 记。"""
    prompt_chars = sum(len(_message_text(m)) for m in messages)
    return {
        "prompt_tokens": max(1, prompt_chars // 2),
        "completion_tokens": max(1, len(output) // 2),
        "prompt_cache_hit_tokens": 0,
    }


class DemoChatModel(BaseChatModel):
    """零外部调用的聊天模型。接口与 ChatOpenAI 对齐到本系统用到的部分。"""

    model_name: str = "demo-mock"
    temperature: float = 0.7
    max_tokens: int = 8192
    streaming: bool = False
    bound_tool_names: Tuple[str, ...] = ()

    @property
    def _llm_type(self) -> str:
        return "demo-mock"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "DemoChatModel":
        """只记录工具名：演示模式不发起工具调用（避免依赖真实向量质量）。"""
        names = tuple(getattr(t, "name", str(t)) for t in tools or [])
        return self.model_copy(update={"bound_tool_names": names})

    def _result(self, messages: Sequence[BaseMessage]) -> ChatResult:
        text = reply_text(messages)
        message = AIMessage(
            content=text,
            response_metadata={
                "model_name": self.model_name,     # 成本表会按这个名字查价 → 查不到 → ¥0
                "token_usage": _estimate_usage(messages, text),
                "finish_reason": "stop",
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._result(messages)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        text = reply_text(messages)
        delay = max(0, int(settings.DEMO_PACE_MS or 0)) / 1000.0 / 4
        for chunk in iter_stream_chunks(text):
            if run_manager:
                run_manager.on_llm_new_token(chunk)
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))
            if delay:
                time.sleep(delay)

    def with_structured_output(
        self,
        schema: Any,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Any:
        """对齐 ChatOpenAI.with_structured_output 的返回形状。

        include_raw=True 时返回 {"raw","parsed","parsing_error"} —— structured.py
        正是按这个形状判断成功与否，形状不对会被误判成失败并降级。
        """
        model = self

        async def _run(messages: Sequence[BaseMessage]) -> Any:
            payload = _schema_payload(schema, messages)
            raw = AIMessage(
                content=json.dumps(payload, ensure_ascii=False),
                response_metadata={
                    "model_name": model.model_name,
                    "token_usage": _estimate_usage(messages, json.dumps(payload)),
                },
            )
            try:
                parsed, error = schema.model_validate(payload), None
            except Exception as exc:  # noqa: BLE001 —— 交给上层三级降级处理
                parsed, error = None, exc
            if include_raw:
                return {"raw": raw, "parsed": parsed, "parsing_error": error}
            if error is not None:
                raise error
            return parsed

        return RunnableLambda(_run)


class DemoHashEmbeddings(Embeddings):
    """确定性哈希向量：不联网，但仍然按字符重合度排序，检索是真的在检索。

    为什么用 md5 而不是 Python 内置 hash()：内置 hash 每次进程启动都加盐，
    灌库进程和查询进程算出的向量不一致，检索会退化成随机。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vector(self, text: str) -> List[float]:
        import hashlib
        import math

        vec = [0.0] * self.dim
        clean = "".join((text or "").split()).lower()
        # 权重取 (0.5, 2.5)：只用二元组会丢失单字线索，只用单字会把"生成"这类
        # 高频字放大到淹没主题（实测 GAN 会被排到第 2，挤掉真正相关的 LLM 文献）。
        for n, weight in ((1, 0.5), (2, 2.5)):
            for i in range(max(0, len(clean) - n + 1)):
                gram = clean[i:i + n]
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dim
                vec[idx] += weight
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)

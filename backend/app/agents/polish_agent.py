# polish_agent.py —— 润色优化 Agent（第 5 个）
import asyncio
import re
from typing import Dict, Any, AsyncIterator, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.agents.base_agent import BaseAgent

# 单块最大字符数。
# 为什么必须分块：实测把 1 万多字的整篇正文交给模型、要求「只输出润色后的完整正文」时，
# 模型会直接原样复制输入（输出与输入逐字相同，相似度 1.0），润色环节完全空转。
# 切成 2500 字左右的块后，模型才会真正逐段改写。
MAX_BLOCK_CHARS = 2500

# 润色输出相对原文的保留比例下限：低于这个值认为是模型把内容写丢了，回退原文该块。
MIN_KEEP_RATIO = 0.4


def _split_by_paragraph(block: str, max_chars: int) -> List[str]:
    """把超长块按空行分段再合并，单段仍超长则硬切。"""
    out: List[str] = []
    buf = ""
    for para in block.split("\n\n"):
        if not para.strip():
            continue
        if not buf:
            buf = para
        elif len(buf) + len(para) + 2 <= max_chars:
            buf = buf + "\n\n" + para
        else:
            out.append(buf)
            buf = para
        # 单段本身就超长（例如没有空行的长段落），直接硬切兜底
        while len(buf) > max_chars:
            out.append(buf[:max_chars])
            buf = buf[max_chars:]
    if buf:
        out.append(buf)
    return out


def split_into_blocks(text: str, max_chars: int = MAX_BLOCK_CHARS) -> List[str]:
    """
    把整篇正文切块。

    优先按 Markdown 二级标题（## 章节）切，再把相邻的小章节贪心合并到接近 max_chars，
    这样既不会把一章拆散，也不会产生十几二十次模型调用（过慢）。
    """
    if not text or len(text) <= max_chars:
        return [text] if text else []

    parts = [p for p in re.split(r"(?m)(?=^##\s)", text) if p.strip()]
    if not parts:
        return _split_by_paragraph(text, max_chars)

    merged: List[str] = []
    buf = ""
    for p in parts:
        if not buf:
            buf = p
        elif len(buf) + len(p) + 2 <= max_chars:
            buf = buf + "\n\n" + p
        else:
            merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)

    blocks: List[str] = []
    for b in merged:
        if len(b) <= max_chars:
            blocks.append(b)
        else:
            blocks.extend(_split_by_paragraph(b, max_chars))
    return blocks or [text]


class PolishAgent(BaseAgent):
    """对论文进行语言、逻辑、规范三个维度的润色优化"""

    def __init__(self):
        super().__init__(
            name="润色专家",
            description="优化论文语言表达和逻辑结构",
        )

    def get_system_prompt(self) -> str:
        return """你是一位资深的学术论文润色专家，擅长优化论文的语言表达和逻辑结构。

润色要点：
1. **语言优化**：消除冗余表达、提升准确性、统一术语、优化句式
2. **逻辑优化**：强化论证链条、完善段落衔接、提升连贯性、优化层次
3. **规范检查**：学术用语规范、格式排版规范、引用格式规范

请保持原文学术观点，仅做表达和结构优化。
必须完整保留原文的所有 Markdown 标题（如 ## 一级章节），标题文字不得改动、不得删减章节。
必须保留原文的 [n] 引用标注，不得新增或删除引用编号。"""

    async def _polish_block(self, text: str, focus_instruction: str) -> str:
        """
        润色单个文本块。

        分块后调用次数变多，单块遇到网络抖动不该让整篇生成失败：
        失败重试一次，仍失败就保留该块原文（降级为"不润色"，绝不丢正文）。
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", """请润色以下论文内容：

{focus_instruction}

## 原文内容：

{content}

## 输出要求：

只输出润色后的正文，不要附加任何修改说明、注释、前言或额外文字。
不要删减任何章节，不要改写标题。

请开始润色："""),
        ])
        chain = prompt | self.llm | StrOutputParser()

        result = ""
        for attempt in range(2):
            try:
                result = await chain.ainvoke({
                    "content": text,
                    "focus_instruction": focus_instruction,
                })
                break
            except Exception:
                if attempt == 1:
                    return text          # 两次都失败 → 保留原文，不影响整体流程
                await asyncio.sleep(2)

        result = (result or "").strip()
        if not result:
            return text
        if len(result) < len(text) * MIN_KEEP_RATIO:
            return text
        return result

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        content = input_data.get("content", "")
        focus = input_data.get("focus", "all")

        # 根据 focus 参数给大模型不同的侧重点
        focus_instructions = {
            "language": "请重点优化语言表达，提升表达的准确性和简洁性。",
            "logic": "请重点优化逻辑结构，强化论证链条和段落衔接。",
            "format": "请重点检查格式规范，确保符合学术写作规范。",
            "all": "请全面优化语言表达、逻辑结构和格式规范。",
        }
        focus_instruction = focus_instructions.get(focus, focus_instructions["all"])

        blocks = split_into_blocks(content)
        if len(blocks) <= 1:
            polished = await self._polish_block(content, focus_instruction) if content else ""
        else:
            polished_blocks = []
            for b in blocks:
                polished_blocks.append(await self._polish_block(b, focus_instruction))
            polished = "\n\n".join(polished_blocks)

        return {
            "polished": polished,
            "original_length": len(content),
            "polished_length": len(polished),
            "blocks": len(blocks),
        }

    async def stream_execute(self, input_data: Dict[str, Any]) -> AsyncIterator[str]:
        content = input_data.get("content", "")
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", f"请润色以下论文内容：\n\n{content}"),
        ])
        chain = prompt | self.llm | StrOutputParser()

        async for chunk in chain.astream({}):
            yield chunk

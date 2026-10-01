# utils/helpers.py —— 通用小工具函数
import json
import re


def parse_json(text: str) -> dict:
    """
    把大模型输出的文本尝试解析成 JSON 对象。
    大模型常会输出带 ```json 包裹的内容，这里自动剥离。
    解析失败时返回空字典（或携带原始文本），避免崩溃。
    """
    if not text:
        return {}
    result = text.strip()
    # 去除 Markdown 代码块包裹，如 ```json ... ```
    if result.startswith("```"):
        lines = result.split("\n")
        result = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {"raw": text}


def to_segments(keywords: str) -> list[str]:
    """把逗号/顿号分隔的关键词字符串拆成列表"""
    if not keywords:
        return []
    return [k.strip() for k in re.split(r"[,，、]", keywords) if k.strip()]
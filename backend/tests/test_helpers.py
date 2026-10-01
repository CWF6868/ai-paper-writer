# 单元测试：utils/helpers.py 的纯函数
from app.utils.helpers import parse_json, to_segments


def test_parse_json_ok():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_fenced():
    """大模型常输出 ```json 包裹的内容，应自动剥离"""
    raw = '```json\n{"score": 0.82, "dimensions": []}\n```'
    data = parse_json(raw)
    assert data["score"] == 0.82
    assert data["dimensions"] == []


def test_parse_json_invalid_returns_raw():
    """非法 JSON 不崩溃，返回带原始文本的 dict"""
    data = parse_json("这不是 JSON")
    assert "raw" in data
    assert data["raw"] == "这不是 JSON"


def test_parse_json_empty():
    assert parse_json(None) == {}
    assert parse_json("") == {}


def test_parse_json_whitespace_only():
    """纯空白按非法 JSON 处理，不崩溃（与现有实现一致：返回 raw）"""
    data = parse_json("   ")
    assert "raw" in data


def test_to_segments_comma():
    assert to_segments("AI,机器学习") == ["AI", "机器学习"]


def test_to_segments_chinese_comma_and_dun():
    """逗号 / 中文逗号 / 顿号混合都要能拆"""
    assert to_segments("A，B、C,D") == ["A", "B", "C", "D"]


def test_to_segments_cleans_whitespace():
    assert to_segments("  AI ,  文本摘要  ") == ["AI", "文本摘要"]


def test_to_segments_empty():
    assert to_segments("") == []
    assert to_segments(None) == []

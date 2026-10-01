# 单元测试：utils/exporter.py —— 重点是 _wrap_cjk 切行（PDF 死循环 bug 的回归测试）
from types import SimpleNamespace

import pytest
from fpdf import FPDF

from app.utils.exporter import (
    _find_cjk_font,
    _plain,
    _ref_lines,
    _wrap_cjk,
    build_docx,
    build_pdf,
)


def _make_pdf():
    font = _find_cjk_font()
    if not font:
        pytest.skip("本机无中文字体，跳过 PDF 相关用例")
    pdf = FPDF()
    pdf.add_font("cjk", "", font)
    pdf.add_page()
    pdf.set_font("cjk", "", 11)
    return pdf


# ---------- _wrap_cjk 回归测试（锁死 fpdf2 wrapmode=CHAR 死循环 bug） ----------

def test_wrap_cjk_short_line():
    """回归：短行（如标题「引言」）曾经让 fpdf2 死循环，现在必须正常返回"""
    pdf = _make_pdf()
    lines = _wrap_cjk(pdf, "引言")
    assert lines == ["引言"]


def test_wrap_cjk_single_char_lines():
    """极端：每个字都超宽的极端场景也不能卡死"""
    pdf = _make_pdf()
    pdf.set_font("cjk", "", 60)  # 超大字号，几乎任何字符都超宽
    lines = _wrap_cjk(pdf, "超宽测试文本")
    assert all(lines)  # 每行非空（逐字兜底）
    assert "".join(lines) == "超宽测试文本"  # 拼接后内容无损


def test_wrap_cjk_mixed_text_reassembles():
    """中英混排：逐行宽度不超可用宽度，且拼接后与原文完全一致"""
    pdf = _make_pdf()
    pdf.set_font("cjk", "", 11)
    text = ("这是一段用于验证中文自动换行的长文本，mixed with English words and 数字123。" * 5)
    lines = _wrap_cjk(pdf, text)
    available = pdf.w - pdf.l_margin - pdf.r_margin
    for line in lines:
        assert pdf.get_string_width(line) <= available + 0.01
    assert "".join(lines) == text


def test_wrap_cjk_respects_newline():
    """含换行符的段落要按行拆分"""
    pdf = _make_pdf()
    lines = _wrap_cjk(pdf, "第一行内容\n第二行内容")
    assert len(lines) == 2
    assert lines[0].startswith("第一行")
    assert lines[1].startswith("第二行")


# ---------- 其它纯函数 ----------

def test_plain_strips_bold_marker():
    assert _plain("**加粗** 和普通") == "加粗 和普通"


def test_ref_lines_format():
    ref = SimpleNamespace(authors="张三", title="论文A", journal="期刊X", year=2020)
    assert _ref_lines([ref]) == ["[1] 张三. 论文A. 期刊X（2020）"]


def test_ref_lines_missing_fields():
    ref = SimpleNamespace(authors=None, title="论文B", journal=None, year=None)
    assert _ref_lines([ref]) == ["[1] 论文B"]


# ---------- 文档生成冒烟 ----------

def _paper():
    return SimpleNamespace(
        title="测试论文标题",
        abstract="这是一段摘要。",
        keywords="AI,论文",
        topic="自然语言处理",
        content="## 引言\n\n这是正文内容，包含 **加粗** 文本。\n\n- 要点一\n- 要点二",
        references=[
            SimpleNamespace(authors="张三", title="论文A", journal="期刊X", year=2020),
        ],
    )


def test_build_docx_bytes():
    data = build_docx(_paper())
    assert data[:2] == b"PK"           # docx 是 zip 容器
    assert len(data) > 1000            # 非空且含正文


def test_build_pdf_bytes():
    pdf = _make_pdf()                  # 顺便探测字体
    del pdf
    data = build_pdf(_paper())
    assert data[:4] == b"%PDF"         # PDF 魔数
    assert len(data) > 500

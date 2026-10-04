# 单元测试：utils/exporter.py
# 重点：_wrap_cjk 切行（fpdf2 wrapmode=CHAR 死循环 bug 的回归）+ docx/pdf 字节冒烟
from io import BytesIO
from types import SimpleNamespace

import pytest
from fpdf import FPDF

from app.utils.exporter import (
    _add_runs,
    _find_cjk_font,
    _markdown_to_pdf,
    _plain,
    _ref_lines,
    _wrap_cjk,
    build_docx,
    build_pdf,
)

CONTENT = "## 引言\n\n这是正文内容，包含 **加粗** 文本。\n\n- 要点一\n- 要点二\n\n###### 六级标题也应正常"


def _make_pdf():
    font = _find_cjk_font()
    if not font:
        pytest.skip("本机无中文字体，跳过 PDF 相关用例")
    pdf = FPDF()
    pdf.add_font("cjk", "", font)
    pdf.add_page()
    pdf.set_font("cjk", "", 11)
    return pdf


def _paper():
    return SimpleNamespace(
        title="测试论文标题",
        abstract="这是一段摘要。",
        keywords="AI,论文",
        topic="自然语言处理",
        content=CONTENT,
        references=[
            SimpleNamespace(authors="张三", title="论文A", journal="期刊X", year=2020),
            SimpleNamespace(authors=None, title="论文B", journal=None, year=None),
        ],
    )


# ---------- _wrap_cjk 回归测试（锁死 fpdf2 死循环 bug） ----------

def test_wrap_cjk_short_line():
    """回归：短行（如标题「引言」）曾经让 fpdf2 死循环，现在必须正常返回"""
    pdf = _make_pdf()
    assert _wrap_cjk(pdf, "引言") == ["引言"]


def test_wrap_cjk_single_char_lines():
    """极端：每个字都超宽的极端场景也不能卡死，且拼接无损"""
    pdf = _make_pdf()
    pdf.set_font("cjk", "", 60)   # 超大字号，几乎任何字符都超宽
    lines = _wrap_cjk(pdf, "超宽测试文本")
    assert all(lines)
    assert "".join(lines) == "超宽测试文本"


def test_wrap_cjk_mixed_text_reassembles():
    """中英混排：逐行宽度不超可用宽度，且拼接后与原文完全一致"""
    pdf = _make_pdf()
    pdf.set_font("cjk", "", 11)
    text = "这是一段用于验证中文自动换行的长文本，mixed with English words and 数字123。" * 5
    lines = _wrap_cjk(pdf, text)
    available = pdf.w - pdf.l_margin - pdf.r_margin
    for line in lines:
        assert pdf.get_string_width(line) <= available + 0.01
    assert "".join(lines) == text


def test_wrap_cjk_respects_newline():
    pdf = _make_pdf()
    lines = _wrap_cjk(pdf, "第一行内容\n第二行内容")
    assert len(lines) == 2
    assert lines[0].startswith("第一行")
    assert lines[1].startswith("第二行")


def test_wrap_cjk_empty_string_yields_one_empty_line():
    pdf = _make_pdf()
    assert _wrap_cjk(pdf, "") == [""]


# ---------- 其它纯函数 ----------

def test_plain_strips_bold_marker():
    assert _plain("**加粗** 和普通") == "加粗 和普通"
    assert _plain("") == ""
    assert _plain(None) == ""


def test_add_runs_splits_bold_and_normal():
    from docx import Document

    doc = Document()
    p = doc.add_paragraph()
    _add_runs(p, "前**中**后")
    assert [r.text for r in p.runs] == ["前", "中", "后"]
    assert [r.bold for r in p.runs] == [None, True, None]


def test_add_runs_without_marker_is_single_run():
    from docx import Document

    doc = Document()
    p = doc.add_paragraph()
    _add_runs(p, "没有加粗")
    assert len(p.runs) == 1


def test_ref_lines_format():
    refs = [SimpleNamespace(authors="张三", title="论文A", journal="期刊X", year=2020)]
    assert _ref_lines(refs) == ["[1] 张三. 论文A. 期刊X（2020）"]


def test_ref_lines_missing_fields():
    refs = [SimpleNamespace(authors=None, title="论文B", journal=None, year=None)]
    assert _ref_lines(refs) == ["[1] 论文B"]


def test_ref_lines_empty():
    assert _ref_lines([]) == []
    assert _ref_lines(None) == []


def test_find_cjk_font_returns_existing_path_or_none():
    font = _find_cjk_font()
    assert font is None or font.endswith(".ttf")


# ---------- Word 文档冒烟 ----------

def test_build_docx_bytes():
    data = build_docx(_paper())
    assert data[:2] == b"PK"           # docx 是 zip 容器
    assert len(data) > 1000


def test_build_docx_contains_structure():
    from docx import Document

    doc = Document(BytesIO(build_docx(_paper())))
    styles = [p.style.name for p in doc.paragraphs]
    assert styles[0] == "Title"                      # 标题
    assert any(s.startswith("Heading") for s in styles)   # markdown 标题被转成 Heading
    assert "List Bullet" in styles                   # 列表项
    bolds = [r.text for p in doc.paragraphs for r in p.runs if r.bold]
    assert "加粗" in bolds
    texts = "\n".join(p.text for p in doc.paragraphs)
    assert "摘要：这是一段摘要。" in texts
    assert "关键词：AI,论文" in texts
    assert "参考文献" in texts
    assert "[1] 张三. 论文A. 期刊X（2020）" in texts


def test_build_docx_tolerates_empty_paper():
    """没有正文 / 文献时也不能崩"""
    empty = SimpleNamespace(title="空论文", abstract=None, keywords=None,
                            topic=None, content="", references=[])
    assert build_docx(empty)[:2] == b"PK"


# ---------- PDF 文档冒烟 ----------

def test_build_pdf_bytes():
    _make_pdf()                        # 顺便探测字体（无字体则 skip）
    data = build_pdf(_paper())
    assert data[:4] == b"%PDF"         # PDF 魔数
    assert len(data) > 500


def test_build_pdf_raises_without_font(monkeypatch):
    import app.utils.exporter as exporter

    monkeypatch.setattr(exporter, "_find_cjk_font", lambda: None)
    with pytest.raises(RuntimeError, match="中文字体"):
        exporter.build_pdf(_paper())


def test_markdown_to_pdf_handles_headings_and_bullets():
    pdf = _make_pdf()
    _markdown_to_pdf(pdf, "# 一级\n## 二级\n### 三级\n#### 四级\n- 项目一\n普通段落\n\n")

    class _P:
        title = "标题"
        abstract = None
        keywords = None
        topic = None
        content = "# 一级\n- 项目一\n普通段落"
        references = []

    assert build_pdf(_P())[:4] == b"%PDF"

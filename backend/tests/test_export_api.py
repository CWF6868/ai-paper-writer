# 集成测试：导出接口（Word / PDF 字节校验）
from app.utils.exporter import _find_cjk_font

CONTENT = "## 引言\n\n这是正文内容，包含 **加粗** 文本。\n\n- 要点一\n- 要点二"


async def _paper_with_content(client, headers):
    r = await client.post("/api/papers/", json={"title": "可导出的论文标题", "topic": "NLP"}, headers=headers)
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    r = await client.put(f"/api/papers/{pid}", json={"content": CONTENT, "status": "completed"}, headers=headers)
    assert r.status_code == 200, r.text
    return pid


async def test_export_docx(client, headers):
    pid = await _paper_with_content(client, headers)
    r = await client.get(f"/api/papers/{pid}/export", params={"format": "docx"}, headers=headers)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.wordprocessingml"
    )
    assert r.content[:2] == b"PK"           # docx 魔数
    assert len(r.content) > 1000
    # 文件名带中文标题
    assert "filename*=UTF-8''" in r.headers["content-disposition"]


async def test_export_pdf(client, headers):
    if not _find_cjk_font():
        import pytest
        pytest.skip("本机无中文字体")
    pid = await _paper_with_content(client, headers)
    r = await client.get(f"/api/papers/{pid}/export", params={"format": "pdf"}, headers=headers)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF"         # PDF 魔数
    assert len(r.content) > 500


async def test_export_bad_format(client, headers):
    pid = await _paper_with_content(client, headers)
    r = await client.get(f"/api/papers/{pid}/export", params={"format": "txt"}, headers=headers)
    assert r.status_code == 400


async def test_export_requires_auth(client):
    assert (await client.get("/api/papers/1/export", params={"format": "docx"})).status_code == 401


async def test_export_cross_user_404(client, user_pair):
    h1, h2 = user_pair
    pid = await _paper_with_content(client, h1)
    r = await client.get(f"/api/papers/{pid}/export", params={"format": "docx"}, headers=h2)
    assert r.status_code == 404

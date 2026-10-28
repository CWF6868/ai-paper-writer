# -*- coding: utf-8 -*-
"""生成 requirements.txt 各包在镜像站的 wheel 直链清单（只查询，不下载）"""
import posixpath
import re
import ssl
import urllib.parse
import urllib.request

REQ = [
    ("fastapi", "0.115.0"),
    ("uvicorn", "0.30.6"),
    ("sqlalchemy", "2.0.35"),
    ("aiosqlite", "0.20.0"),
    ("pydantic", "2.9.2"),
    ("pydantic-settings", "2.5.2"),
    ("langchain", "0.3.7"),
    ("langchain-community", "0.3.7"),
    ("langchain-openai", "0.2.10"),
    ("langgraph", "0.2.57"),
    ("chromadb", "0.5.15"),
    ("posthog", "2.4.2"),
    ("PyJWT", "2.9.0"),
    ("passlib", "1.7.4"),
    ("bcrypt", "4.0.1"),
    ("python-multipart", "0.0.12"),
    ("email-validator", "2.2.0"),
    ("python-docx", "1.1.2"),
    ("fpdf2", "2.8.9"),
    # chromadb 的硬依赖，Windows 上最容易失败的那个
    ("chroma-hnswlib", "0.7.6"),
]

DEV = [
    ("pytest", "8.3.3"),
    ("pytest-asyncio", "0.24.0"),
    ("httpx", "0.27.2"),
    ("pytest-cov", "5.0.0"),
]

MIRRORS = [
    ("清华", "https://pypi.tuna.tsinghua.edu.cn/simple/{pkg}/"),
    ("阿里", "https://mirrors.aliyun.com/pypi/simple/{pkg}/"),
]

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def norm(s):
    return re.sub(r"[._-]", "", s).lower()


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pip/26.0"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
        return r.read().decode("utf-8", "ignore")


def best_link(base, html, pkg, ver):
    """返回 (文件名, 绝对直链)"""
    want_pkg, want_ver = norm(pkg), norm(ver)
    hits = []
    for href in re.findall(r'href="([^"]+)"', html):
        path = href.split("#")[0]
        base_name = posixpath.basename(urllib.parse.urlparse(path).path)
        if base_name.endswith(".whl"):
            parts = base_name[:-4].split("-")
            if len(parts) >= 5 and norm(parts[0]) == want_pkg and norm(parts[1]) == want_ver:
                hits.append((base_name, path))
    if not hits:
        return None, None
    win = [h for h in hits if "win_amd64" in h[0] and ("cp311" in h[0] or "abi3" in h[0])]
    pure = [h for h in hits if "none-any" in h[0]]
    pick = (win or pure)[0]
    return pick[0], urllib.parse.urljoin(base, pick[1])


def build(pkgs, title, lines):
    lines.append(f"## {title}\n")
    lines.append("| 包 | 版本 | wheel 文件 | 清华镜像直链 | 阿里镜像直链 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for pkg, ver in pkgs:
        row = {}
        for label, tmpl in MIRRORS:
            base = tmpl.format(pkg=pkg)
            try:
                row[label] = best_link(base, fetch(base), pkg, ver)
            except Exception as e:
                row[label] = (None, f"查询失败 {type(e).__name__}")
        fname = next((v[0] for v in row.values() if v[0]), "-")
        tuna = row["清华"][1] or "-"
        ali = row["阿里"][1] or "-"
        lines.append(f"| {pkg} | {ver} | `{fname}` | {tuna} | {ali} |")
    lines.append("")


def main():
    lines = [
        "# 后端依赖镜像下载清单（Python 3.11 / Windows x64）",
        "",
        "生成方式：直接读取镜像站 simple 索引，只挑选 cp311-win_amd64 或纯 Python 的 wheel。",
        "所有链接均来自官方镜像，供网络不稳定时手动下载后离线安装。",
        "",
        "## 一键安装命令（推荐，先试这个）",
        "",
        "```bat",
        "cd /d D:\\ai-paper-writer-main\\backend",
        "venv\\Scripts\\activate",
        "python -m pip install -U pip setuptools wheel",
        "pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple "
        "--trusted-host pypi.tuna.tsinghua.edu.cn --timeout 120 --retries 10",
        "```",
        "",
        "## 离线安装（网络反复中断时）",
        "",
        "```bat",
        "cd /d D:\\ai-paper-writer-main\\backend",
        "venv\\Scripts\\activate",
        "pip download -r requirements.txt -d wheels "
        "-i https://pypi.tuna.tsinghua.edu.cn/simple --only-binary :all: "
        "--timeout 120 --retries 20",
        "pip install --no-index --find-links wheels -r requirements.txt",
        "```",
        "",
    ]
    build(REQ, "生产依赖（requirements.txt）+ chromadb 关键依赖", lines)
    build(DEV, "测试依赖（requirements-dev.txt，非运行必需）", lines)
    lines += [
        "## 说明",
        "",
        "- `chroma-hnswlib==0.7.6` 是 chromadb 的硬依赖，Windows 上必须有 cp311 的 wheel；",
        "  Python 3.13 没有该 wheel，会退化成源码编译并因缺少 MSVC 而失败，这是最常见的翻车原因。",
        "- 纯 Python wheel（`-py3-none-any.whl`）与平台无关，任何机器都能装。",
        "- 若某个直链失效，用同目录下的索引页查看：",
        "  `https://pypi.tuna.tsinghua.edu.cn/simple/<包名>/`",
        "",
    ]
    out = "\n".join(lines)
    with open("依赖镜像下载清单.md", "w", encoding="utf-8") as f:
        f.write(out)
    print(out[:2000])
    print("\n...\n已写入 backend/依赖镜像下载清单.md")


if __name__ == "__main__":
    main()

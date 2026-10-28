# -*- coding: utf-8 -*-
"""核对 requirements.txt 各包在镜像站是否有适配 cp311-win_amd64 的 wheel（诊断用，不安装）"""
import posixpath
import re
import ssl
import sys
import urllib.parse
import urllib.request

PKGS = [
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
]

BASE = {
    "tuna": "https://pypi.tuna.tsinghua.edu.cn/simple/{pkg}/",
    "aliyun": "https://mirrors.aliyun.com/pypi/simple/{pkg}/",
    "ustc": "https://mirrors.ustc.edu.cn/pypi/simple/{pkg}/",
}

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def norm(s):
    return re.sub(r"[._-]", "", s).lower()


def parse_filename(href):
    """从 href 解析出 (规范化包名, 版本, 文件名)"""
    path = href.split("#")[0]
    base = posixpath.basename(urllib.parse.urlparse(path).path or path)
    if base.endswith(".whl"):
        parts = base[:-4].split("-")
        if len(parts) >= 5:
            return norm(parts[0]), norm(parts[1]), base
    elif base.endswith((".tar.gz", ".zip")):
        stem = base[:-7] if base.endswith(".tar.gz") else base[:-4]
        parts = stem.split("-")
        if len(parts) >= 2:
            return norm(parts[0]), norm(parts[1]), base
    return None, None, base


def pick(links, pkg, ver):
    want_pkg, want_ver = norm(pkg), norm(ver)
    cands = []
    for href in links:
        n, v, base = parse_filename(href)
        if n == want_pkg and v == want_ver:
            cands.append((base, href))
    if not cands:
        return "MISS", "该版本不在索引中", ""
    win = [c for c in cands if "win_amd64" in c[0] and ("cp311" in c[0] or "abi3" in c[0])]
    pure = [c for c in cands if "none-any" in c[0]]
    sdist = [c for c in cands if c[0].endswith((".tar.gz", ".zip"))]
    for group, label in ((win, "Windows 专用 wheel 就绪"), (pure, "纯 Python wheel 就绪")):
        if group:
            base, href = group[0]
            full = urllib.parse.urljoin(BASE_URL, href) if not href.startswith("http") else href
            return "OK", f"{label}  |  {base}", full
    if sdist:
        base, href = sdist[0]
        full = urllib.parse.urljoin(BASE_URL, href) if not href.startswith("http") else href
        return "SRC", f"仅源码包，需编译  |  {base}", full
    base, href = cands[0]
    return "WARN", f"无适配 cp311-win_amd64 的 wheel  |  {base}", ""


def main():
    global BASE_URL
    mirror = sys.argv[1] if len(sys.argv) > 1 else "tuna"
    BASE_URL = BASE[mirror].format(pkg="")
    print(f"镜像: {BASE[mirror].format(pkg='<pkg>')}\n")
    rows, problems = [], []
    for pkg, ver in PKGS:
        url = BASE[mirror].format(pkg=pkg)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pip/26.0"})
            with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                html = r.read().decode("utf-8", "ignore")
        except Exception as e:
            rows.append((pkg, ver, "FAIL", f"索引访问失败 {type(e).__name__}", ""))
            problems.append(rows[-1])
            continue
        st, msg, link = pick(re.findall(r'href="([^"]+)"', html), pkg, ver)
        rows.append((pkg, ver, st, msg, link))
        if st != "OK":
            problems.append(rows[-1])

    for pkg, ver, st, msg, _ in rows:
        print(f"[{st:4}] {pkg}=={ver}  {msg}")
    print("\n===== 需要关注 =====")
    if not problems:
        print("全部 19 个直接依赖在该镜像都有可用于 Python 3.11 / Windows 的 wheel")
    for pkg, ver, st, msg, link in problems:
        print(f"- {pkg}=={ver} [{st}] {msg}")
        if link:
            print(f"  {link}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""核对关键传递依赖在镜像上是否有 cp311-win_amd64 wheel（不限版本，取最高可用版）"""
import posixpath
import re
import ssl
import urllib.parse
import urllib.request

# 这些是 chromadb / uvicorn / langchain 链条上需要二进制或在 Windows 上易翻车的包
PKGS = [
    "chroma-hnswlib",
    "onnxruntime",
    "tokenizers",
    "numpy",
    "pydantic-core",
    "httptools",
    "watchfiles",
    "websockets",
    "mmh3",
    "grpcio",
    "orjson",
    "greenlet",
    "tiktoken",
    "tqdm",
    "httpx",
    "openai",
    "PyYAML",
    "pypika",
    "overrides",
    "importlib-resources",
    "typing-extensions",
    "python-dotenv",
    "colorama",
    "SQLAlchemy",
    "anyio",
    "starlette",
    "tenacity",
    "requests",
    "aiohttp",
    "dataclasses-json",
]

BASE = "https://pypi.tuna.tsinghua.edu.cn/simple/{pkg}/"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def vkey(v):
    parts = re.split(r"[._-]", v)
    out = []
    for p in parts:
        out.append((0, int(p)) if p.isdigit() else (1, p))
    return out


def classify(base):
    if base.endswith(".whl"):
        tags = base[:-4].split("-")[2:]
        tagstr = "-".join(tags)
        if "win_amd64" in tagstr and ("cp311" in tagstr or "abi3" in tagstr or "cp36" in tagstr):
            return "WIN"
        if "none-any" in tagstr:
            return "PURE"
        return "OTHER"
    if base.endswith((".tar.gz", ".zip")):
        return "SRC"
    return "OTHER"


def main():
    print(f"{'包':22} {'最高可用版本':14} {'状态':5} 文件名")
    print("-" * 100)
    bad = []
    for pkg in PKGS:
        try:
            req = urllib.request.Request(BASE.format(pkg=pkg), headers={"User-Agent": "pip/26.0"})
            with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                html = r.read().decode("utf-8", "ignore")
        except Exception as e:
            print(f"{pkg:22} {'-':14} FAIL  索引访问失败 {type(e).__name__}")
            bad.append((pkg, "索引访问失败"))
            continue
        files = []
        for href in re.findall(r'href="([^"]+)"', html):
            base = posixpath.basename(urllib.parse.urlparse(href.split("#")[0]).path)
            if base.endswith(".whl"):
                parts = base[:-4].split("-")
                if len(parts) >= 5:
                    files.append((parts[1], base, classify(base)))
            elif base.endswith(".tar.gz"):
                stem = base[:-7]
                parts = stem.split("-")
                if len(parts) >= 2:
                    files.append((parts[-1], base, "SRC"))
        if not files:
            print(f"{pkg:22} {'-':14} MISS  索引里无文件")
            bad.append((pkg, "索引里无文件"))
            continue
        good = [f for f in files if f[2] in ("WIN", "PURE")]
        if good:
            best = sorted(good, key=lambda f: vkey(f[0]))[-1]
            print(f"{pkg:22} {best[0]:14} {best[2]:5} {best[1]}")
        else:
            only_src = [f for f in files if f[2] == "SRC"]
            print(f"{pkg:22} {'-':14} SRC   仅源码，无 Windows wheel")
            bad.append((pkg, "仅源码包，需要 Visual C++ 编译环境"))
    print("\n===== 需要关注 =====")
    if not bad:
        print("全部有可用 wheel")
    for pkg, why in bad:
        print(f"- {pkg}: {why}")


if __name__ == "__main__":
    main()

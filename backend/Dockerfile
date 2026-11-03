# syntax=docker/dockerfile:1.6
# backend/Dockerfile —— 多阶段构建：builder 出 wheel，runtime 只装运行时
#
# 为什么分阶段：chromadb 依赖 onnxruntime / tokenizers 等带编译产物的包，
# 直接在 runtime 镜像里 pip install 会把 gcc、头文件、pip 缓存全留在最终镜像里
# （实测多出 400MB+）。这里只在 builder 里有编译工具链，runtime 用 wheel 离线安装。
#
# 构建源可覆盖（默认走官方源，便于他人复现；国内构建建议传镜像，否则 apt/pip 可能慢到超时）：
#   docker compose build \
#     --build-arg APT_MIRROR=http://mirrors.aliyun.com \
#     --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple

# ---------- Stage 1: 构建 wheel ----------
FROM python:3.11-slim AS builder

ARG APT_MIRROR=""
ARG PIP_INDEX_URL="https://pypi.org/simple"

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /build

# gcc/g++ 只在少数包没有预编译 wheel 时需要；装在 builder 里，不进最终镜像
RUN set -eux; \
    if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null \
        || sed -i "s|http://deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && pip wheel --wheel-dir /wheels -r requirements.txt


# ---------- Stage 2: 运行时 ----------
FROM python:3.11-slim AS runtime

ARG APT_MIRROR=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 路径必须落在 /app/data（compose 里的命名卷），否则容器重建即丢库
ENV DATABASE_URL="sqlite:////app/data/paper_writer.db" \
    VECTOR_DB_PATH="/app/data/chroma"

# 中文字体：PDF 导出依赖系统字体，缺了会抛「未找到可用的中文字体」。
# wqy 体积约 20MB，比 noto-cjk（~200MB）小一个数量级，够用。
RUN set -eux; \
    if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null \
        || sed -i "s|http://deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends fonts-wqy-zenhei fonts-wqy-microhei; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt ./
# 运行期用不到的东西一起清掉：pip/setuptools、标准库测试套件、编译缓存，
# 合计约 50MB；不清的话镜像会压在 1.2GB 验收线上。
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels \
 && rm -rf /usr/local/lib/python3.11/test \
           /usr/local/lib/python3.11/tkinter \
           /usr/local/lib/python3.11/idlelib \
           /usr/local/lib/python3.11/ensurepip \
           /usr/local/lib/python3.11/lib2to3 \
           /usr/local/lib/python3.11/pydoc_data \
           /usr/local/lib/python3.11/site-packages/pip* \
           /usr/local/lib/python3.11/site-packages/setuptools* \
           /usr/local/lib/python3.11/site-packages/wheel* \
 && find /usr/local/lib/python3.11 -name "__pycache__" -type d -prune -exec rm -rf {} +

# 只拷运行需要的代码：tests / venv / 本地 data 由 .dockerignore 挡掉
COPY app ./app
COPY scripts ./scripts

# 非 root 运行：容器被拿下时不该顺带拿到 root
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/data \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 健康检查用标准库，不额外装 curl（少一层攻击面、少 ~5MB）
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status == 200 else 1)"

# 单 worker：SQLite + 本地 Chroma 都是进程内状态，多 worker 会各写各的，
# 表现为"刚生成的论文刷新就没了"。要扩并发得先换 Postgres + 外部向量库。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

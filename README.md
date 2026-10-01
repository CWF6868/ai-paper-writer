# AI 论文写作系统

基于**多 Agent 协作 + RAG 检索增强**的智能论文写作平台。输入选题后，系统自动完成大纲规划、分章节写作、质量评审与迭代重写，并支持论文的 Word / PDF / Markdown 导出。

## 功能特性

- **一键生成论文**：输入选题即可自动生成完整论文，支持大纲、摘要、章节正文、参考文献
- **多 Agent 协作流程**：选题 Agent → 规划 Agent → 写作 Agent → 评审 Agent 流水线协作，评审不达标自动重写（最多 2 轮）
- **SSE 流式输出**：生成过程实时推送进度，前端逐段展示
- **RAG 文献增强**：内置 42 篇学术文献向量库（阿里云百炼 text-embedding-v3 + Chroma），写作时检索引用相关文献
- **质量评审**：从多个维度自动打分并给出修改意见，评分低于阈值触发重写循环
- **论文管理**：创建 / 编辑 / 删除论文，按状态（草稿 / 生成中 / 已完成）筛选，关键词搜索
- **多格式导出**：Word（.docx）、PDF（中文字体适配）、Markdown
- **用户认证**：JWT 登录注册，论文数据按用户隔离

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 后端 | FastAPI · SQLAlchemy · SQLite · LangGraph · LangChain · Chroma |
| LLM | DeepSeek（DeepSeek-chat，OpenAI 兼容接口） |
| 向量化 | 阿里云百炼 text-embedding-v3 |
| 前端 | Vue 3 · Vite · Vue Router · Axios |
| 测试 | pytest · pytest-asyncio · httpx |

## 系统架构

```
用户输入选题
    │
    ▼
┌─────────────────────────── 后端流水线 ───────────────────────────┐
│ 选题推荐 Agent ─► 大纲规划 Agent ─► 章节写作 Agent ─► 评审 Agent │
│        ▲                                │              │        │
│        │                                ▼              ▼        │
│        └───── 质量不达标时重写（最多 2 轮）◄── 质量评分/意见      │
│                                                                │
│  RAG：写作时从 Chroma 向量库检索相关文献（42 篇，text-embedding-v3）│
└────────────────────────────────────────────────────────────────┘
    │
    ▼
SSE 流式推送生成进度 ─► Vue3 前端实时展示 ─► 导出 Word / PDF / Markdown
```

## 目录结构

```
├── backend/                  # FastAPI 后端
│   ├── app/
│   │   ├── agents/           # 各角色 Agent（选题/规划/写作/评审）
│   │   ├── workflows/        # LangGraph 多 Agent 编排流程
│   │   ├── rag/              # RAG 向量库（embedding + Chroma 检索）
│   │   ├── api/              # 路由（auth / papers / topics / export）
│   │   ├── services/         # 业务服务层
│   │   ├── models/           # SQLAlchemy 数据模型
│   │   ├── schemas/          # Pydantic 校验模型
│   │   ├── core/ utils/      # 配置 / 工具
│   │   ├── config.py         # 环境变量配置
│   │   ├── database.py       # 数据库连接
│   │   └── main.py           # 应用入口（lifespan 自动建表/初始化向量库）
│   ├── tests/                # pytest 测试
│   ├── seed_rag.py           # 一次性：初始化 42 篇文献向量库
│   └── requirements.txt      # 生产依赖
├── frontend/                 # Vue3 前端
│   └── src/
│       ├── views/            # 页面（论文列表 / 详情 / 生成 / 登录）
│       ├── components/       # 组件
│       ├── api/              # Axios 封装（相对路径 /api）
│       └── router/ utils/
└── .env.example              # 环境变量模板
```

## 快速开始

### 1. 后端

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# 复制环境变量模板并填写 API Key
cp .env.example .env

# 启动（首次启动自动建表并初始化向量库）
uvicorn app.main:app --reload --port 8000
```

### 2. 初始化 RAG 文献库（仅首次）

```bash
cd backend
python seed_rag.py   # 灌入 42 篇文献，之后重启后端生效
```

### 3. 前端

```bash
cd frontend
npm install
npm run dev          # 开发服务器，/api 自动代理到 8000
```

访问 `http://localhost:5173`，注册账号后即可开始使用。

## 运行测试

```bash
cd backend
pytest                # 需要先安装 requirements-dev.txt
```

## API 摘要

| 模块 | 端点 | 说明 |
| --- | --- | --- |
| 认证 | `POST /api/auth/register` `POST /api/auth/login` `GET /api/auth/me` | 注册 / 登录 / 当前用户 |
| 论文 | `GET/POST /api/papers` `GET/PUT/DELETE /api/papers/{id}` | 论文 CRUD |
| 生成 | `POST /api/papers/{id}/generate` | 同步生成 |
| 生成 | `POST /api/papers/{id}/stream-generate` | **SSE 流式生成**（推荐） |
| 导出 | `GET /api/papers/{id}/export?format=docx\|pdf\|md` | 多格式导出 |
| 选题 | `POST /api/topics/recommend` | 选题推荐 |

## 环境变量

所有配置通过 `backend/.env` 提供，关键项：

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek API Key（LLM 写作大脑） |
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key（文献向量化） |
| `SECRET_KEY` | JWT 签名密钥，生产环境务必更换 |
| `DATABASE_URL` | 数据库连接串，默认 SQLite |
| `VECTOR_DB_PATH` | Chroma 向量库路径 |

完整模板见 [`backend/.env.example`](backend/.env.example)。

## License

[MIT](LICENSE)

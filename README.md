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

## P0-1 结构化输出 + 工具调用

评审/大纲路径已从「字符串约定 + 静默降级」升级为 Schema 结构化输出（Pydantic 出口类型恒定）+ 显式工具调用，核心设计：

- **Schema 即接口**（`app/llm/schemas.py`）：5 维评审打分固定枚举、评分锚点写进字段描述、综合分与维度均值自动对账（容差 0.03）、`evidence` 强制可溯源、`severity`/`must_fix` 让重写有靶子。
- **三级降级**（`app/llm/structured.py`）：function_calling → json_mode → 裸文本解析，每一级记录结构化 JSON 观测日志（`app.llm.observability`，字段与 llm_call_log/agent_tool_log 口径对齐，表结构留到 P0-3 成本归因一起做）；全失败显式抛 `StructuredOutputError`。
- **工具调用**（`app/llm/tools/`）：只挂 2 个「LLM 自己产不出信息」的工具——`verify_citation`（引用查证：embedding 余弦 + 关键词重合确定性初筛，仅 [0.35, 0.72] 灰区升级 LLM 裁决，约 80% 查证零 LLM 往返）、`search_literature`（当前论文参考文献内的语料语义检索，编号与查证工具同源）。预算硬上限：最多 2 轮工具往返 / 6 次调用（`registry.py`，被 `test_tool_budget` 锁定）。
- **不编假分数**：评审不可用（异常/三级全失败/越界分）→ `quality_score=None + quality_degraded=True`，放行但不假装合格；历史假 0.8 分数污染评测的路径已删除。降级标记一路落库（`papers.quality_degraded`，启动时自动补列迁移）并随详情接口、同步生成响应、SSE `done` 事件下发，前端据此显示「质量评审未完成」而非把评审块静默隐藏——**「没有分数」必须让用户看见，不能让它等同于「没有评审」**。
- **观测确实可见**（`app/__init__.py` + `app/utils/logging_config.py`）：项目原先没有任何 logging 配置，Python lastResort handler 只输出 WARNING+ 且**不渲染 extra** —— 结果所有 `LLM_CALL`/`TOOL_CALL` 的 INFO 日志被静默丢弃、降级原因一起丢失（2026-11-01 评测实测：6 篇全跑，观测日志 0 条）。现在包导入即配置日志，`LOG_LEVEL` 环境变量可调，降级原因写进 message 而非 extra。

### 验收标准

| 指标 | 改造前 | 改造后 |
|---|---|---|
| 评审/大纲 Agent 的 `parse_json` 调用 | 2 处（主解析路径） | **0 处**（`grep app/agents -e parse_json` 可验证，仅注释提及） |
| 静默假分数（评审挂 → 0.8） | 存在 | **0**，降级显式标记 `quality_degraded` |
| 降级态前端呈现 | 评审块静默消失（`v-if="qualityScore != null"`） | 显示「质量评审未完成」警示块 + 重新生成入口；`quality_degraded` 落库，随详情接口与 SSE `done` 事件下发 |
| 评审分数越界处理 | 手写 `max(0.0, min(1.0, ...))` | Pydantic `ge/le` 在边界拦截 |
| 章节名匹配 | 正则模糊匹配为主 | Schema「逐字抄写」约束为主，`_norm_section_name` 降为兜底 |
| 工具调用预算 | — | **≤ 2 轮 / ≤ 6 次**（单测锁定） |
| 观测日志可见性 | 无 logging 配置，INFO 全丢 | 包入口 `app/__init__.py` 统一 `setup_logging`，INFO 可见 |
| 模型输出上限 | 未设置（落到 4096 默认值） | `LLM_MAX_TOKENS=8192`，且经 `extra_body` 透传为 `max_tokens`（langchain-openai 会把它改写成 DeepSeek 不认的 `max_completion_tokens`，实测直接失效） |
| 测试 | 216 | **239**（含 schema/降级链/灰区/预算/收口清洗/并发隔离/形状漂移/分制漂移/阈值标定 22 项新防线） |

### 端到端实测（最新：2026-11-02，6 篇真实生成 + 首检分观测）

这一轮的目的是补上「重写前的首检分」——此前只记录最终分，导致质量阈值只能靠猜。

| id | 首检分 | 重写后分 | 增益 |
|---|---|---|---|
| 1 | 0.44 | 0.51 | +0.07 |
| 2 | 0.49 | 0.51 | +0.02 |
| 3 | 0.44 | 0.44 | 0.00 |
| 4 | 0.44 | 0.44 | 0.00 |
| 5 | 0.42 | 0.44 | +0.02 |
| 6 | 0.42 | 0.42 | 0.00 |

首检分：中位数 0.44、均值 0.442、标准差 0.026、极差 0.07；重写平均增益 +0.018。

**阈值标定结论**（`scripts/analyze_threshold.py` 按实测数据推演每个阈值的后果）：

| 阈值 | 重写率 | 预估平均质量 | 说明 |
|---|---|---|---|
| ≤ 0.42 | 0% | 0.442 | 门禁失效：全部直接放行 |
| **0.43–0.44** | **33%** | **0.445** | **当前取值 0.44（实测中位数）** |
| 0.45–0.49 | 83% | 0.457 | 仅最高分那篇能通过 |
| ≥ 0.50 | 100% | 0.460 | 门禁退化为固定流程 |

取 0.44：质量差异（-0.015）在评分噪声带（0.02）内，换来重写率从 100% 降到 33%，门禁第一次真正具备「放行」能力。若更看重质量，可把 `.env` 的 `QUALITY_PASS_THRESHOLD` 调到 0.45~0.49（重写率 83%）。

**这一轮评测又抓出 4 个真实缺陷**（均已修复并锁进测试）：

1. **`max_tokens` 未设置**（最严重）：不显式设置时落到 DeepSeek 默认 4096，评审要输出 5 个维度的长点评+原文证据，JSON 参数被截断成非法 JSON → 三级降级全失败 → 整篇评审作废。实测第 2 篇因此降级。现固定 `settings.LLM_MAX_TOKENS=8192`。
   > **2026-11-02 复查补充**：只写 `max_tokens=8192` 参数其实**没生效**。langchain-openai 0.2.10 会把该参数改写成 `max_completion_tokens` 再发请求，而 DeepSeek 兼容接口不认这个字段、静默忽略（对照实测：`max_tokens=16` → `finish_reason=length`、只出 47 字符；`max_completion_tokens=16` → `finish_reason=stop`、照常输出 246 tokens）。现改为 `extra_body={"max_tokens": limit}` 透传，并由 `test_get_llm_sends_max_tokens_via_extra_body` 锁住「请求里必须没有 `max_completion_tokens`」。此前截断现象消失，实际是评审输出变短所致，不是这个配置起了作用。
2. **10 分制漂移**：模型偶尔按 10 分制打分（`score=2.2`、`overall_score=6.5`），0-1 校验直接判死。现在「全部分数都 >1 且 ≤10」时归一化，个别字段越界仍拒绝（不能靠除以 10 把垃圾洗成合法值）。
3. **外层包装/字段别名**：模型给出 `{"review": {"overall_score": …, "summary": …}}`，现自动拆包并映射 `overall_score→score`、`summary→overall`。
4. 补跑工具 `scripts/eval_one.py`：单篇失败只补跑该篇，不必整轮重跑。

### 端到端实测（2026-11-01，6 篇真实生成）

| 指标 | 改造前（10-27 基线） | 改造后（11-01） |
|---|---|---|
| 评审降级篇数 | 0（假分数掩盖） | **0 / 6**（修复 3 个降级 bug 后） |
| 平均质量分 | 0.5733 | 0.4783 |
| 分数标准差 | 0.0734 | 0.0426 |
| 重写触发率 | 100% | 100% |
| 单篇平均耗时 | 119.0s | 204.2s（+71%，工具查证 9 次/篇） |
| 平均字数 | 8984 | 9767 |

**结论（如实记录）**：

1. **「拉开分数堆积」这个指标没有达成**。改造后标准差从 0.073 收到 0.043，极差 0.20 → 0.09，分数更集中。原因是 6 篇均为同一流水线、同一模板产出的同质论文，质量本就接近——分数接近是**评分正确的表现**，不是锚点失效。11-02 的首检分实测（0.42–0.49，标准差 0.026）再次印证：瓶颈在生成质量的同质性，不在评分尺度。
2. **锚点本身有效**。用构造样本做区分度探针（`scripts/probe_score_spread.py`）：结构完整、有指标与消融的论文得 0.44，空洞套话论文得 0.22（2 倍差距），且评审意见具体可核（指出「12.3%」与「12.3 个百分点」口径矛盾、缺显著性检验、相关工作仅 3 篇即下断言）。

> 复现命令：`python scripts/eval_generate.py --out data/eval_generate_v4.json`；阈值标定：`python scripts/analyze_threshold.py`；单篇补跑：`python scripts/eval_one.py 2`；区分度探针：`python scripts/probe_score_spread.py`。

### 已知优化点

- **动态章节名枚举（方案 B）**：评审 Schema 评估过 `Literal[titles]` 动态枚举（章节名在生成时约束成固定选项，`_norm_section_name` 可彻底删除），因 schema 体积随章节数膨胀、且大纲标题常带「第3章」前缀污染枚举值、DeepSeek 对复杂 schema 遵循度下降，选择不用，保留「Schema 约束 + 归一化兜底」（方案 A）。
- **观测落表**：`llm_call_log` 表已建成（P0-3），token / 缓存命中 / 成本 / 延迟按节点与用户归因，聚合接口见 `GET /api/metrics/llm`。

## P0-2 并发 + 异步化

三处浪费，按金额排序：**章节串行**（8 章 × 6–10s，章节之间零依赖却排队写）、**假异步检索**（`async def` 里同步调 embedding HTTP + Chroma）、**导出阻塞事件循环**（`build_pdf` 实测 636ms，导出期间整个进程的 SSE 进度僵住）。

### 改了什么

| 位置 | 改动 |
|---|---|
| `graph_builder._write_sections` | 串行 `for`+`await` → `asyncio.gather` + `Semaphore(4)`（`SECTION_CONCURRENCY` 可配） |
| 同上：顺序 | 结果**按大纲下标回填**，不按完成顺序（`sections_content` 插入序 = 成文序，`_polish_paper` 拼接与 `order_index` 都依赖它） |
| 同上：失败 | `return_exceptions=True` 收齐后按章报错：`章节写作失败（1/4 章）：「第2章」ValueError: ...`，`CancelledError` 原样上抛 |
| `rag/retriever.search` | `to_thread(vector_store.search, ...)` |
| `api/export` | `to_thread(build_docx / build_pdf, ...)`，派发前强制加载 `references` 防止线程内惰性加载 |
| `utils/llm.get_llm` | `max_retries=3`（并发抬高后偶发 429 的概率上升，不重试则一次限流整章报废） |

### 实测（2026-11-02）

| 口径 | 改造前 | 改造后 | 变化 |
|---|---|---|---|
| **同一篇、同 2 轮重写**（题目1） | 205.3s | **129.3s** | **−37.0%** |
| **每章次耗时**（÷ 章节数 × 质检次数，归一化） | 13.8s | **8.8s** | **−35.7%** |
| 6 篇整体（含结构性差异，见下） | 210.2s | 76.9s | −63.4% |
| 单机吞吐（论文级并发 1，stub 基准） | 5.35 篇/分 | **11.64 篇/分** | 2.18× |
| 单机吞吐（论文级并发 4） | 16.05 篇/分 | **34.87 篇/分** | 2.17× |
| 写作段内部（8 章 / 并发 4） | 8.0s | **2.0s** | 4.00× |
| 平均质量分 | 0.450 | 0.468 | 在 ±0.02 噪声带内 |

### 三句必须说清的话

1. **−63.4% 不等于并发收益**。V4 是 10:36 跑的，当时判定硬编码 `>= 0.7`，6 篇首检分（0.42–0.52）全不达标 → 100% 重写；V5 用 0.44 阈值 → 5/6 篇一次过。少写一轮等于少写 7–8 章。**并发的净效果看同条件 −37.0% 与归一化 −35.7%（两个独立口径互证）。**
2. **`to_thread` 不加速单次导出**。python-docx / fpdf2 是 CPU 密集型，GIL 挡着，收益只有「导出期间事件循环仍可调度」（对照实测：同步阻塞 200ms → tick 0 次；`to_thread` → tick 14 次）。
3. **为什么不是 4 倍**。用题目1 两轮数据反解：`16w + F = 205.3`、`4w + F = 129.3` → `w ≈ 6.3s/章`、`F ≈ 104s`。约 104s 的固定段（大纲 / 润色 / 评审，全部依赖全文，不可并发化）决定了单篇下限。

> 复现：`python scripts/bench_concurrency.py --tag after --papers 6 --levels 1,2,4`（stub，不耗额度）；`python scripts/eval_generate.py --out data/eval_generate_v5.json`；`python scripts/compare_eval.py --before data/eval_generate_v4.json --after data/eval_generate_v5.json`。
> 设计细节与取舍：[`docs/P0-2-并发与异步化设计.md`](docs/P0-2-并发与异步化设计.md)。

## P0-3 LLM 调用治理（成本）

改造前成本完全不可见：没有 token 记录、没有节点归因、不知道缓存命中多少、更算不出"这篇论文花了多少钱"。

### 计价的三条事实（先纠正口径，否则数字全错）

1. 配置里的 `deepseek-chat` **已于 2026-07-24 下线**，仅作别名，实测响应 `model_name` 是 `deepseek-flash` → **按响应里的模型名归一化计价**，不能按配置值查表（否则查不到价、成本静默记 0）。
2. **峰谷双价**（2026-08-17 生效）：北京时间工作日 09–12、14–18 为峰时，单价是空闲的 2 倍。同一 prompt 上午跑比晚上贵一倍。
3. **缓存命中价仅为未命中的 1/30**，必须单独计量 `cache_hit_tokens`，否则成本虚高且看不出缓存的价值。

### 改了什么

| 位置 | 改动 |
|---|---|
| `core/llm_client.py`（新） | 单价表（缓存命中/未命中/输出 × 峰谷）、模型别名归一、`compute_cost`（Decimal 量化 1e-8）、`UsageRecord`、`extract_usage`、`CostTrackingCallback`、`llm_scope`、`capture_loop`、`build_chat_model` 唯一构造出口 |
| `utils/llm.py` | 收敛为薄壳，避免超时/回调分叉 |
| `workflows/graph_builder.py` | `_metered` 包装 7 个节点打 `node/paper_id/user_id` 标签，不动业务方法体 |
| `models/llm_call_log.py`（新） | `cost` 用 `Numeric(18,8)`（单次常在 1e-5 元，Float 累加会漂）；**不加外键**（论文删了流水要留存）；`user_id` 列支撑按用户归因 |
| `core/llm_log_sink.py`（新） | 非阻塞入队 + 每 0.5s / 50 条批量提交 + 哨兵停止 + 跨线程投递 + 对账 `stats()` |
| `api/metrics.py`（新） | `GET /api/metrics/llm`：按节点 / 天 / 模型聚合 token、成本、延迟 |

### 实测（2026-11-02，真实 DeepSeek）

| 指标 | 值 |
|---|---|
| **单篇成本** | **0.0863 元**（三次独立测量区间 0.076–0.095） |
| 调用次数 / token | 18 次｜输入 28,423 / 输出 12,346 / 缓存命中 8,192（28.8%） |
| 缓存省下的钱 | 0.0119 元（成本的 **13.8%**） |
| 错峰价值 | 同样 token 全在峰时要 0.1726 元 → 空闲时段省 **50%** |
| 成本结构 | 输出 token 占 **64.4%**，输入未命中 35.2%，输入命中 0.5% |
| 计量覆盖率 | **100%**（独立计数源 22/22） |
| 落库开销 | 0.5 μs/次，一篇 22 次调用合计 **≈ 11 μs** |
| 落库对账 | 18 收到 / 18 写入 / 0 丢弃，`balanced=True` |

### 三句必须说清的话

1. **成本由"生成多少字"决定**，不是"读了多少"。输出单价是输入未命中的 3 倍，而一次重写 = 多写 7–8 章 ≈ 成本翻倍。P0-1 把通过线从硬编码 `0.7` 调到 `0.44`，5/6 篇不再重写——**省的不只是时间，也是一半的钱**。
2. **缓存有天花板**：它只作用于输入。命中率做到 100% 也只能省约 34%，救不了输出成本。
3. **同一个坑踩了两次**：回调看似是异步的（继承 `AsyncCallbackHandler` 并实现 `aon_*`），实测（langchain-core 0.3.63）`ainvoke` 下真正调用的仍是**同步** `on_llm_end`，且在线程池 `asyncio_0` 里执行。第一次靠 `start()` 绑定 loop 侥幸没露馅，第二次走无 lifespan 的运行方式就整批丢了 16 条。现在 loop 由 `llm_scope` 里的 `capture_loop()` 记住，不依赖 `start()`。

> 复现：`python scripts/report_llm_cost.py`（真实生成并出账单）；`--skip-generate` 只读已有数据。
> 设计细节、单价口径分歧与遗留风险：[`docs/P0-3-LLM成本治理设计.md`](docs/P0-3-LLM成本治理设计.md)。

## API 摘要

| 模块 | 端点 | 说明 |
| --- | --- | --- |
| 认证 | `POST /api/auth/register` `POST /api/auth/login` `GET /api/auth/me` | 注册 / 登录 / 当前用户 |
| 论文 | `GET/POST /api/papers` `GET/PUT/DELETE /api/papers/{id}` | 论文 CRUD |
| 生成 | `POST /api/papers/{id}/generate` | 同步生成 |
| 生成 | `POST /api/papers/{id}/stream-generate` | **SSE 流式生成**（推荐） |
| 导出 | `GET /api/papers/{id}/export?format=docx\|pdf\|md` | 多格式导出 |
| 选题 | `POST /api/topics/recommend` | 选题推荐 |
| 成本 | `GET /api/metrics/llm?days=7&node=&user_id=` | LLM 调用成本/延迟聚合（普通用户仅自己，管理员可全量） |

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

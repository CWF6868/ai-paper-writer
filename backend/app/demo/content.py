# demo/content.py —— 演示模式下的确定性内容生成
#
# 设计边界（很重要）：演示模式**只替换模型输出**，不替换任何业务逻辑。
# RAG 检索、章节受控并发、质量门禁、SSE 推流、Word/PDF 导出全部走生产代码，
# 所以演示页上看到的分数、章节数、引用编号都是算出来的，不是写死的假数据。
#
# 为什么内容要确定性（同输入 → 同输出）：
#   1) 录演示 GIF 时不能每次跑出不同的字数和分数，否则脚本对不上；
#   2) 公网 Demo 被面试官点开时，表现必须稳定，不能"这次好下次崩"。
import hashlib
import re

# ---------------------------------------------------------------------------
# 大纲：固定的学术论文结构，字数按总字数等比分配
# ---------------------------------------------------------------------------
_SECTION_PLAN = [
    ("摘要", ["研究背景与问题定义", "方法与主要贡献", "主要结果与结论"], 600),
    ("引言", ["研究问题的现实背景", "现有方法的局限", "本文的研究思路与贡献"], 900),
    ("相关工作", ["大语言模型的文本生成能力", "检索增强生成的研究脉络", "现有工作的不足"], 900),
    ("方法", ["整体框架设计", "检索与证据注入策略", "生成与质量门控机制"], 1200),
    ("实验与分析", ["实验设置与数据来源", "主要结果对比", "消融与误差分析"], 1200),
    ("结论", ["主要结论", "研究局限", "未来工作方向"], 600),
]


def _seed(text: str) -> int:
    """稳定的整数种子：用 md5 而不是 hash()，后者每个进程加盐、重启就变。"""
    return int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "big")


def build_outline(prompt: str) -> dict:
    """返回符合 OutlineResult Schema 的 dict。入参可以是纯标题，也可以是整段 Prompt。"""
    title_m = re.search(r"标题：\s*(.+)", prompt or "")
    title = title_m.group(1).strip() if title_m else (prompt or "").strip()[:80]
    title = title or "未命名研究"
    sections = [
        {"title": name, "points": list(points), "word_count": words}
        for name, points, words in _SECTION_PLAN
    ]
    return {"title": title, "sections": sections}


# ---------------------------------------------------------------------------
# 章节正文
# ---------------------------------------------------------------------------
_SECTION_RE = re.compile(r"##\s*章节标题：(.+)")
_POINTS_RE = re.compile(r"###\s*内容要点：\n(.*?)\n\n###", re.S)
_REFS_RE = re.compile(r"###\s*相关文献[^\n]*：\n(.*?)\n\n###", re.S)

_OPENERS = {
    "摘要": "本文围绕「{title}」展开，关注的是生成结果的证据支撑与可核查性问题。",
    "引言": "「{title}」这一问题的现实背景在于：生成式方法已经能产出通顺文本，但结论的可信度仍缺少稳定保障。",
    "相关工作": "回顾「{title}」相关的研究脉络，可以看到两条相对独立的技术路线。",
    "方法": "针对「{title}」，本文把整体方案拆成检索、注入与门控三个可独立验证的环节。",
    "实验与分析": "为检验「{title}」的有效性，本文在统一设置下对比了几组基线方案。",
    "结论": "围绕「{title}」的研究可以得到三点相对稳健的结论。",
}

_DEFAULT_OPENER = [
    "围绕「{title}」这一问题，既有研究已从方法设计与实证检验两条路径给出解释，但在适用边界上仍留有讨论空间。",
    "「{title}」是本文论证链条中的关键环节，其结论直接影响后续分析的可信度。",
]

_CONNECTORS = ["首先", "其次", "进一步", "在此基础上"]

_EXPANSIONS = [
    "该判断成立的前提是数据分布与训练阶段保持一致，一旦部署环境发生偏移，结论就需要重新标定。",
    "其可行性依赖两个条件：中间结果可被完整记录，以及评价口径在多次比较中保持不变。",
    "这一环节在工程上通常表现为额外的校验步骤，代价是延迟上升，收益是错误可以被定位。",
    "若缺少这一环节，系统依然能产出结果，但结果的可解释性与可复现性都会明显下降。",
    "已有工作报告了方向一致的现象，不过这些工作多在较小规模的数据上完成验证。",
]


def _extract_section_prompt(prompt: str) -> tuple:
    """从写作 Prompt 里解析出（标题, 要点列表, 引用编号列表）。

    解析失败不抛错：宁可生成一段通用内容，也不让演示流程中断。
    """
    title_m = _SECTION_RE.search(prompt)
    title = title_m.group(1).strip() if title_m else "研究内容"

    points_m = _POINTS_RE.search(prompt)
    points = []
    if points_m:
        points = [
            line.lstrip("-").strip()
            for line in points_m.group(1).splitlines()
            if line.strip().startswith("-")
        ]

    refs_m = _REFS_RE.search(prompt)
    refs_text = refs_m.group(1) if refs_m else ""
    # 只认 [n] 形式的编号，且必须真的出现在文献列表里，避免编造引用
    available = {int(n) for n in re.findall(r"\[(\d+)\]", refs_text)}
    citations = sorted(available)[:3]
    return title, points, citations


def build_section(prompt: str) -> str:
    """按章节标题 + 要点生成正文；有文献编号时按编号引用，没有就不编。"""
    title, points, citations = _extract_section_prompt(prompt)
    key = next((k for k in _OPENERS if k in title), None)
    if key:
        opener = _OPENERS[key].format(title=title)
    else:
        opener = _DEFAULT_OPENER[_seed(title) % len(_DEFAULT_OPENER)].format(title=title)
    paragraphs = [opener]

    for i, point in enumerate(points[:4]):
        connector = _CONNECTORS[i % len(_CONNECTORS)]
        # 用「序号 + 标题种子」选扩写句：同一节内相邻要点不会撞同一句
        # （只按 point 做种子时实测出现过两句一模一样的收尾，一眼看出是模板）
        expansion = _EXPANSIONS[(i + _seed(title)) % len(_EXPANSIONS)]
        paragraphs.append(f"{connector}，{point}。{expansion}")

    first_point = points[0] if points else title
    paragraphs.append(
        f"把上述几点放在一起看，「{title}」的关键并不在于单点技术的选择，"
        f"而在于各环节之间的接口是否稳定：只有「{first_point}」这一环的输入质量可控，"
        f"后续环节才可能得到可比较的输出。"
    )

    if citations:
        refs = "、".join(f"[{n}]" for n in citations)
        paragraphs.append(
            f"上述判断与已有文献的观察方向一致 {refs}。"
            f"需要说明的是，这些结论大多建立在特定语料与特定评价口径之上，"
            f"直接外推到其他场景时仍需谨慎。"
        )
    else:
        paragraphs.append(
            "需要说明的是，当前可用的公开证据仍不充分，相关结论应视为待验证的假设，"
            "而不是已经确立的事实。"
        )

    paragraphs.append(
        f"基于以上讨论，本节把「{title}」拆解为可验证的子问题，"
        f"并给出对应的分析路径，为后续章节的展开提供接口。"
    )
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# 评审结果
# ---------------------------------------------------------------------------
_DIMENSIONS = [
    "创新性与学术价值",
    "逻辑结构与论证",
    "内容完整度与充分性",
    "语言表达与学术规范",
    "文献引用质量",
]


def _parse_review_input(prompt: str) -> tuple:
    """从评审 Prompt 中取（标题, 正文, 章标题列表）。"""
    title_m = re.search(r"##\s*论文标题\n(.+)", prompt)
    title = title_m.group(1).strip() if title_m else "未命名论文"
    body_m = re.search(r"##\s*正文内容\n(.*?)\n\n##\s*参考文献", prompt, re.S)
    body = body_m.group(1) if body_m else prompt
    headings = [h.strip() for h in re.findall(r"(?m)^##\s+(.+)$", body)]
    refs_m = re.search(r"##\s*参考文献\n(.*)$", prompt, re.S)
    has_refs = bool(re.search(r"\[\d+\]", refs_m.group(1))) if refs_m else False
    return title, body, headings, has_refs


def build_review(prompt: str) -> dict:
    """返回符合 ReviewResult Schema 的 dict。

    分数刻意落在 0.62-0.72：高于默认门禁 0.44，演示一遍过、不触发重写循环；
    但仍留出"未达发表水平"的评语空间，不会造出一个 0.9 的假优秀。
    """
    title, body, headings, has_refs = _parse_review_input(prompt)
    base = 0.62 + (_seed(title) % 9) / 100.0        # 0.62 - 0.70

    scores = [round(base + delta, 2) for delta in (0.02, 0.0, -0.03, 0.01, 0.0)]
    if not has_refs:
        scores[4] = 0.45                            # 没有参考文献，引用维度必须低
    if len(body) < 800:
        scores[2] = round(scores[2] - 0.05, 2)      # 内容偏短，充分性扣分

    dims = [
        {
            "dimension": name,
            "score": score,
            "comment": comment,
            "evidence": "",
        }
        for name, score, comment in zip(
            _DIMENSIONS,
            scores,
            [
                "选题与既有工作的差异点交代得还不够尖锐，创新性主张需要更明确的对照。",
                "章节之间的过渡基本成立，但论证链条在中段出现了断点。",
                "核心论点有支撑，部分小节缺少数据或案例的进一步展开。",
                "术语使用基本统一，个别长句可以拆分以提升可读性。",
                "引用编号与文末列表一致，但引用密度在不同章节间不均衡。",
            ],
        )
    ]

    section_feedback = [
        {
            "section": h,
            "advice": "建议补充该章与前后章节的衔接句，并把结论收敛到一句话。",
            "severity": "major",
            "evidence": "",
        }
        for h in headings[:2]
    ]

    return {
        "score": round(sum(scores) / len(scores), 2),
        "dimensions": dims,
        "section_feedback": section_feedback,
        # must_fix 留空：演示模式不触发重写循环，避免 3 分钟演示变成 8 分钟
        "must_fix": [],
        "overall": (
            "文章结构完整、论述方向清晰，具备进一步打磨的基础；"
            "主要问题集中在创新点的表述强度与中段论证的支撑材料上。"
        ),
    }


def build_citation_verdict(prompt: str) -> dict:
    """CitationVerdict 的演示裁决：灰区一律判 partial，符合"不过度猜"的约束。"""
    sim_m = re.search(r"相似度\s*([0-9.]+)", prompt)
    sim = float(sim_m.group(1)) if sim_m else 0.5
    verdict = "supported" if sim >= 0.72 else ("unsupported" if sim < 0.35 else "partial")
    return {"verdict": verdict, "similarity": round(sim, 3), "reason": "演示模式：按向量相似度直接裁决。"}


# ---------------------------------------------------------------------------
# 其他节点的文本输出
# ---------------------------------------------------------------------------
def build_topic_list(prompt: str) -> str:
    """选题推荐：保留用户的研究领域，给出 3 个结构完整的选题。"""
    field_m = re.search(r"研究领域：(.+)", prompt)
    field = field_m.group(1).strip() if field_m else "该研究方向"
    return f"""## 选题推荐

### 选题 1：面向{field}的证据约束生成方法研究
**研究方向**：生成过程中的证据对齐
**研究价值**：降低生成内容与事实不一致的风险，提升可核查性
**创新点**：把检索证据的覆盖度纳入生成目标，而不只做事后校验
**可行性分析**：数据与算力要求可控，评测指标已有公开基线

### 选题 2：{field}中的多阶段质量门控机制
**研究方向**：生成质量控制
**研究价值**：在不显著增加成本的前提下稳定产出质量
**创新点**：以可解释的维度分数替代单一总分门禁
**可行性分析**：可在现有流水线上增量改造，风险较低

### 选题 3：面向{field}的检索增强效果归因分析
**研究方向**：检索与生成的贡献拆解
**研究价值**：回答"增益来自检索还是来自提示"这一常见疑问
**创新点**：设计可分离的消融实验框架
**可行性分析**：实验成本低，适合作为第一阶段工作"""


def build_reference_list(prompt: str) -> str:
    """文献整理兜底：从 Prompt 里已有的文献信息拼出编号列表，不新增任何文献。"""
    titles = re.findall(r"标题:\s*(.+)", prompt)
    if not titles:
        return "暂无参考文献"
    return "\n".join(f"[{i}] {t.strip()}[J]." for i, t in enumerate(titles, start=1))


def build_generic(prompt: str) -> str:
    """兜底输出：任何未识别的节点都给一段可读文本，不让流程因为空串中断。"""
    return (
        "演示模式说明：当前实例未连接真实大模型，以下内容由确定性规则生成，"
        "用于验证完整流程（检索 → 写作 → 评审 → 导出）可以跑通。"
    )

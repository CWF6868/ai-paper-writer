# demo/content.py —— 演示模式下的确定性内容生成
#
# 设计边界（很重要）：演示模式**只替换模型输出**，不替换任何业务逻辑。
# RAG 检索、章节受控并发、质量门禁、SSE 推流、Word/PDF 导出全部走生产代码，
# 所以演示页上看到的分数、章节数、引用编号都是算出来的，不是写死的假数据。
#
# 两条必须同时守住的性质：
#   1) 确定性（同输入 → 同输出）：录演示 GIF 时字数与分数不能每次抖，否则脚本对不上；
#   2) 区分度（不同选题 → 不同产出）：早期版本用一套固定章节 + 固定要点 +
#      按章节名取种子的句式，结果「唐代边塞诗」和「RAG 质量门控」生成逐字相同的正文，
#      演示反而证明系统是假的。现在章节骨架按学科分派，术语与论据从题目槽位派生。
#
# 区分度靠三样东西，都不需要中文分词：
#   - 标题结构模式：「面向X的Y」「X中的Y」「X对Y的影响」——中文论文标题的稳定特征；
#   - 学科线索词：决定用哪套章节骨架（理工 / 人文 / 医学 / 综述）；
#   - 种子来源：章节正文的句式种子 = 论文题目 + 章节名，跨篇不撞车。
import hashlib
import random
import re

# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def _seed(*parts: str) -> int:
    """稳定的整数种子：用 md5 而不是 hash()，后者每个进程加盐、重启就变。"""
    return int.from_bytes(hashlib.md5("|".join(parts).encode("utf-8")).digest()[:4], "big")


# ---------------------------------------------------------------------------
# 题目槽位：把标题拆成「研究对象」与「限定范围」
#
# 为什么不用分词：为了演示模式引入 jieba 这类依赖不值当，而且中文论文标题
# 有很稳定的结构模式。实测按模式抽取的准确率明显高于「按领域词切分」
# （后者会把「社区高血压患者用药依从性干预」切成「社区高血压患/者用药依/从性干」）。
# ---------------------------------------------------------------------------
_TITLE_PATTERNS = [
    (r"^面向(.+?)的(.+?)(?:研究|探析|初探|$)", ("scope", "topic")),
    (r"^基于(.+?)的(.+?)(?:研究|探析|初探|$)", ("method", "topic")),
    (r"^(.+?)在(.+?)中的应用(?:研究|$)", ("topic", "scope")),
    (r"^(.+?)中的(.+?)(?:研究|探析|初探|$)", ("scope", "topic")),
    (r"^(.+?)对(.+?)的影响(?:研究|$)", ("iv", "dv")),
    (r"^(.+?)与(.+?)的关系(?:研究|$)", ("a", "b")),
    (r"^(.+?)(?:研究|探析|初探)$", ("topic", "")),
]


def title_slots(title: str) -> dict:
    """返回 {"topic": 研究对象, "scope": 限定范围}。抽不到就按长度对半劈。"""
    text = (title or "").strip().strip("《》").strip()
    if not text:
        return {"topic": "该主题", "scope": "相关情境"}

    got = None
    for pattern, names in _TITLE_PATTERNS:
        m = re.match(pattern, text)
        if m:
            got = dict(zip(names, [g.strip() for g in m.groups()]))
            break

    if got is None:
        got = {"topic": text}

    topic = got.get("topic") or got.get("dv") or got.get("b") or text
    scope = got.get("scope") or got.get("iv") or got.get("a") or got.get("method") or ""

    # 无模式可依且标题够长：对半劈，至少让「限定范围」有点内容
    if not scope and len(topic) >= 10:
        half = len(topic) // 2
        scope, topic = topic[:half], topic[half:]

    topic = topic[:16].strip() or text[:16]
    scope = scope[:16].strip() or topic
    return {"topic": topic, "scope": scope}


# ---------------------------------------------------------------------------
# 学科判定：决定章节骨架
# ---------------------------------------------------------------------------
_DOMAIN_CUES = {
    "cs": ["大模型", "语言模型", "检索", "生成", "算法", "深度学习", "神经网络", "推荐",
           "知识图谱", "多模态", "RAG", "Agent", "模型", "智能", "数据", "文本", "质量", "门控"],
    "human": ["诗", "文学", "历史", "哲学", "社会", "文化", "教育", "政策", "语言", "传播",
              "唐", "宋", "意象", "叙事", "美学", "边塞", "地理", "文本"],
    "med": ["临床", "患者", "药物", "病理", "诊断", "疗效", "细胞", "基因", "队列", "治疗",
            "依从", "干预", "血压", "糖尿病", "随机对照"],
    "review": ["综述", "研究进展", "文献计量", "前沿", "趋势", "述评"],
}


def detect_domain(title: str, topic: str = "", paper_type: str = "") -> str:
    """按题目线索判定学科；论文类型是综述时优先走综述骨架。"""
    ptype = (paper_type or "").lower()
    if "review" in ptype or "综述" in (paper_type or ""):
        return "review"
    text = f"{title or ''}{topic or ''}"
    scores = {k: sum(1 for w in v if w in text) for k, v in _DOMAIN_CUES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "cs"


# ---------------------------------------------------------------------------
# 章节骨架：每套骨架与该学科的真实论文结构对齐
# 第三项是篇幅权重，用来把总字数分配到各章
# ---------------------------------------------------------------------------
_PLANS = {
    "cs": [
        ("摘要", ["研究背景与问题定义", "方法与主要贡献", "主要结果"], 0.12),
        ("引言", ["{topic}在真实场景中的需求", "现有方案在{scope}上的局限", "本文的切入点"], 0.15),
        ("相关工作", ["{topic}的技术脉络", "{scope}方向的代表性工作", "尚未解决的问题"], 0.15),
        ("方法：{topic}的实现框架", ["整体设计", "{scope}的注入与约束策略", "质量门控与失败回退"], 0.24),
        ("实验与分析", ["数据与评价口径", "与基线在{scope}指标上的对比", "消融与误差来源"], 0.22),
        ("结论", ["主要结论", "适用边界", "后续工作"], 0.12),
    ],
    "human": [
        ("摘要", ["研究对象与问题", "材料与方法", "主要判断"], 0.12),
        ("引言", ["{topic}研究的既有分歧", "本文要回答的问题", "研究路径说明"], 0.15),
        ("概念界定与文献回顾", ["{topic}的界定差异", "既有解释框架", "本文采用的界定"], 0.17),
        ("{topic}的类型与文本表现", ["材料范围与选取标准", "典型文本的细读", "类型之间的过渡"], 0.26),
        ("历史语境中的{scope}", ["{scope}的制度与观念背景", "文本与语境的对应关系", "例外情况"], 0.18),
        ("结论与余论", ["主要判断", "材料的局限", "可延伸的问题"], 0.12),
    ],
    "med": [
        ("摘要", ["研究目的", "方法", "主要结果与结论"], 0.12),
        ("引言", ["{topic}的疾病负担", "现有证据的不足", "本研究假设"], 0.15),
        ("资料与方法", ["研究对象与纳排标准", "干预与对照设置", "统计方法"], 0.21),
        ("结果", ["基线特征", "主要终点", "{topic}的亚组与安全性"], 0.28),
        ("讨论", ["与既往研究的一致性", "机制层面的解释", "局限性"], 0.14),
        ("结论", ["临床意义", "适用人群", "后续验证"], 0.10),
    ],
    "review": [
        ("摘要", ["综述范围", "纳入文献概况", "主要发现"], 0.12),
        ("引言", ["{topic}领域的发展脉络", "开展本综述的必要性", "综述问题"], 0.15),
        ("检索策略与文献概况", ["数据库与检索式", "纳入排除流程", "文献计量特征"], 0.22),
        ("研究主题聚类", ["主题一：{topic}的效果证据", "主题二：{scope}的机制解释", "主题间的交叉"], 0.27),
        ("争议与研究空白", ["尚未收敛的分歧", "方法学局限", "可推进的方向"], 0.14),
        ("展望", ["理论层面", "实践层面"], 0.10),
    ],
}

def _plan_for(domain: str) -> list:
    return _PLANS.get(domain) or _PLANS["cs"]


def build_outline(prompt: str) -> dict:
    """返回符合 OutlineResult Schema 的 dict。入参可以是纯标题，也可以是整段 Prompt。"""
    title_m = re.search(r"标题：\s*(.+)", prompt or "")
    title = title_m.group(1).strip() if title_m else (prompt or "").strip()[:80]
    title = title or "未命名研究"

    type_m = re.search(r"论文类型：\s*(\S+)", prompt or "")
    paper_type = type_m.group(1) if type_m else ""

    limit_m = re.search(r"总字数要求：\s*约?\s*(\d+)", prompt or "")
    word_limit = int(limit_m.group(1)) if limit_m else 6000

    slots = title_slots(title)
    plan = _plan_for(detect_domain(title, "", paper_type))

    sections = []
    for name, points, weight in plan:
        sections.append({
            "title": name.format(**slots),
            "points": [p.format(**slots) for p in points],
            # 字数按权重分配；下限 300，避免出现「80 字的章节」这种一眼假的分配
            "word_count": max(300, int(word_limit * weight)),
        })
    return {"title": title, "sections": sections}


# ---------------------------------------------------------------------------
# 章节正文
# ---------------------------------------------------------------------------
_SECTION_RE = re.compile(r"##\s*章节标题：(.+)")
_TITLE_RE = re.compile(r"##\s*论文题目：(.+)")
_POINTS_RE = re.compile(r"###\s*内容要点：\n(.*?)\n\n###", re.S)
_REFS_RE = re.compile(r"###\s*相关文献[^\n]*：\n(.*?)\n\n###", re.S)
_WORDS_RE = re.compile(r"预估字数：\s*(\d+)")

# 章节角色：决定开头句与收尾句。
#
# 为什么不用「哈希取模选模板」：一套 3 句的开头表配 6 个章节，必然有章节撞同一句
# （实测摘要与引言开头逐字相同）。按角色选就不会：一篇文章里每个角色只出现一次。
_ROLE_HINTS = [
    ("摘要", ("摘要", "Abstract")),
    ("引言", ("引言", "绪论")),
    ("相关工作", ("相关", "概念", "界定", "检索", "聚类", "空白", "综述")),
    ("方法", ("方法", "资料", "设计")),
    ("结果", ("结果", "实验", "分析")),
    ("讨论", ("讨论",)),
    ("结论", ("结论", "余论", "展望")),
]

_ROLE_OPENERS = {
    "摘要": [
        "本文围绕「{title}」展开，讨论{topic}在{scope}中的表现与边界。",
        "本文以「{title}」为研究对象，重点考察{topic}在{scope}情境中的表现与边界。",
    ],
    "引言": [
        "「{title}」值得讨论，是因为{topic}的既有结论在不同材料与口径下并不一致。",
        "就「{title}」而言，{topic}已经有相当数量的研究，但结论仍存在分歧。",
    ],
    "相关工作": [
        "回顾{topic}的研究脉络，可以看到两条相对独立的路径。",
        "围绕{topic}，已有研究在{scope}方向上积累了可观的证据，但彼此可比性有限。",
    ],
    "方法": [
        "本节说明{topic}的处理方式：先把问题拆成可独立检验的环节，再逐项给出操作定义。",
        "针对{topic}在{scope}上的不足，本节给出具体的处理方案与判据。",
    ],
    "结果": [
        "按预设口径，{topic}相关指标呈现出若干值得注意的差异。",
        "结果部分先给出总体特征，再按{topic}的关键维度逐项展开。",
    ],
    "讨论": [
        "上述结果与既往关于{topic}的报道既有吻合之处，也存在需要解释的偏差。",
        "{topic}的观察结果需要放回{scope}的语境中理解，否则容易过度解释。",
    ],
    "结论": [
        "围绕「{title}」的讨论可以收敛为三点判断。",
        "本文对{topic}的讨论可归纳为以下结论。",
    ],
    "其他": [
        "讨论「{title}」的{section}，需要先厘清概念边界，再落到可检验的命题上。",
        "就「{title}」而言，{topic}是理解{section}的前提。",
    ],
}

_ROLE_CLOSINGS = {
    "摘要": "以上是本文关于{topic}的核心主张与主要依据，具体论证见后续各节。",
    "引言": "下文按「问题—方法—证据」的顺序展开对{topic}的判断。",
    "相关工作": "上述关于{topic}的工作构成本文的起点，也留下了本节末尾指出的空白。",
    "方法": "上述针对{topic}的设计，其参数与实现细节将在实验部分给出。",
    "结果": "这些关于{topic}的结果构成下一节讨论的事实基础。",
    "讨论": "上述对{topic}的解释仍需更多证据支持，这构成本文的主要局限之一。",
    "结论": "至此，本文关于{topic}的主要工作与不足均已交代。",
    "其他": "本节对{section}的判断将作为下一节讨论的前提。",
}


def _role_for(section: str) -> str:
    for role, hints in _ROLE_HINTS:
        if any(h in section for h in hints):
            return role
    return "其他"

# 要点承接句：不能写成「首先，X。」这种电报体，要带上承接关系
_LEADINS = [
    "首先，就{point}而言",
    "其次需要说明的是{point}",
    "进一步看{point}",
    "在此基础上，{point}同样需要交代",
    "最后，{point}这一环节不能省略",
]

# 论据句：按「观察 + 限定」两段拼装，而不是从固定句表里挑。
#
# 为什么不用固定句表：一套 8 句的表在 6 个章节里抽 20 多次，同一篇论文内部
# 必然重复（实测每句在一篇里出现 2-3 次），读者一眼看出是模板。
# 拆成「6 条观察 × 6 条限定」后组合数是 36，同一篇内基本不撞同一句。
_CLAIMS = {
    "cs": [
        "该设计成立的前提是训练与部署的数据分布保持一致",
        "代价主要体现在推理延迟与显存占用上，收益是错误可以被定位到具体环节",
        "在基线对比中，这一改动带来的增益随数据规模上升而收窄",
        "缺少这层约束时系统仍能产出结果，但同一输入两次运行的差异会明显变大",
        "这一环节在工程上表现为额外的校验步骤：先记录中间产物，再按固定口径比对",
        "把评价口径从单一指标换成多指标加权后，部分结论会发生反转",
    ],
    "human": [
        "这一判断依赖材料的代表性，孤证不足以支撑整体结论",
        "同一表述在不同文本中承担的功能并不相同",
        "既有研究多依据选集类文献，这可能放大某些特征的出现频率",
        "若把时间范围放宽到前后各半个世纪，这一类型的边界会明显变得模糊",
        "这类表达往往通过意象的并置完成，其意义在并置关系中被生产出来",
        "编纂与流传过程本身会筛掉一部分材料，现存文本未必等于当时的写作面貌",
    ],
    "med": [
        "该结果对纳入标准较为敏感，放宽年龄上限后效应量有所下降",
        "组间差异在亚组分析中并未保持一致",
        "随访时间偏短，长期安全性与依从性的持续性仍缺少数据",
        "样本量限制了亚组分析的检验效能",
        "主要分析采用意向性治疗原则，缺失值以多重填补处理",
        "与既往研究相比，本研究的效应量偏小，可能与入组人群的疾病严重程度较低有关",
    ],
    "review": [
        "不同研究采用的评价工具不一致，直接合并结果存在风险",
        "纳入文献多为单中心设计，外部效度有限",
        "发表偏倚难以完全排除，阴性结果可能被系统性低估",
        "近三年的研究开始从效果验证转向机制解释",
        "检索策略覆盖了主流数据库，但灰色文献与会议论文的纳入有限",
        "各研究的干预时长差异较大，从两周到一年不等",
    ],
}

_QUALIFIERS = {
    "cs": [
        "一旦部署环境发生偏移，阈值就需要重新标定",
        "因此结论应限定在可复现的实验条件下",
        "这说明它更像补丁而非根本解法",
        "可复现性随之下降，排查成本明显上升",
        "这一顺序本身会引入额外延迟，需要在调度层做补偿",
        "口径的选择因此需要显式声明，否则跨研究无法比较",
    ],
    "human": [
        "需要更多同类文本交叉印证才能下判断",
        "脱离上下文单看一句，容易把修辞误读为事实陈述",
        "使边缘案例被系统性忽略",
        "断代问题会直接影响结论的稳定性",
        "这提示研究单位应当是段落而非单句",
        "对存世材料的统计因此只能视为下限估计",
    ],
    "med": [
        "提示结论的适用人群需要限定",
        "可能存在效应修饰因素，不宜直接推广到全部患者",
        "阴性结果应理解为「未发现差异」而非「确认无效」",
        "敏感性分析用于检验该结论是否稳健",
        "合并用药的分布差异可能对结局产生混杂影响",
        "因此需要更大样本的前瞻研究加以验证",
    ],
    "review": [
        "更适合做叙述性综合而非定量合并",
        "跨地区推广时需要重新验证",
        "这会放大整体效应的乐观程度",
        "但证据强度仍不足以支撑因果推断",
        "可能遗漏部分早期工作",
        "使「效果能否持续」这一问题难以回答",
    ],
}


def _extract_section_prompt(prompt: str) -> dict:
    """从写作 Prompt 里解析出论文题目、章节名、要点、引用编号与预估字数。

    解析失败不抛错：宁可生成一段通用内容，也不让演示流程中断。
    """
    paper_m = _TITLE_RE.search(prompt)
    paper_title = paper_m.group(1).strip() if paper_m else ""

    title_m = _SECTION_RE.search(prompt)
    section = title_m.group(1).strip() if title_m else "研究内容"

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
    available = sorted({int(n) for n in re.findall(r"\[(\d+)\]", refs_text)})

    words_m = _WORDS_RE.search(prompt)
    word_count = int(words_m.group(1)) if words_m else 800

    return {
        "paper_title": paper_title,
        "section": section,
        "points": points,
        "citations": available,
        "word_count": word_count,
    }


def _evidence(domain: str, seed_key: int, k: int) -> str:
    """拼一段「观察+限定；观察+限定」的双句论据。

    单句段落只有 40-60 字，凑够篇幅要十几段，读起来像要点罗列；
    双句段落约 110-120 字，更接近学术写作的实际段落密度。
    两句各取不同的观察与限定（k 按不同步长推进），组合空间是 6^4，
    同一篇论文内基本不会整段重复——这是「不同论文长得不一样」之外的第二道保险。
    """
    claims = _CLAIMS[domain]
    quals = _QUALIFIERS[domain]
    a = (seed_key // 3 + k) % len(claims)
    b = (seed_key // 5 + k * 2) % len(claims)
    if b == a:                       # 两半撞同一条观察时错开一位，避免同段自我重复
        b = (a + 1) % len(claims)
    qa = quals[(seed_key // 11 + k * 5) % len(quals)]
    qb = quals[(seed_key // 13 + k * 3 + 1) % len(quals)]
    if qb == qa:                    # 同一段里两半用同一条限定，读起来像复读
        qb = quals[(quals.index(qa) + 1) % len(quals)]
    return f"{claims[a]}，{qa}；{claims[b]}，{qb}。"


_ROLE_ORDER = ["摘要", "引言", "相关工作", "方法", "结果", "讨论", "结论", "其他"]

# 引用句与无引用句各给 6 个版本，按章节角色取。
# 只给一个版本时，6 个章节会出现 4-6 段完全相同的句子（实测），一眼看出是模板。
_CITE_LINES = [
    "上述关于{topic}的判断与已有工作的观察方向一致 {refs}，但在外推到其他材料与口径时仍需谨慎。",
    "在{topic}上，本文与{refs}的观察方向相同，差别主要在材料范围与评价口径。",
    "这些关于{topic}的结论在{refs}中也有报告，不过那些研究多在更小的样本上完成验证。",
    "就{refs}而言，本文关于{topic}的结论与其并不冲突，但对其适用边界作了更窄的限定。",
    "上述对{topic}的分析与{refs}的判断可以互相印证，分歧集中在效应量的大小上。",
    "把本文关于{topic}的结果与{refs}放在一起看，方向一致而细节存在差异，提示口径需要统一。",
]

_NO_CITE_LINES = [
    "就{topic}而言，当前可用的公开证据仍不充分，相关结论应视为待验证的假设，而不是已经确立的事实。",
    "这一节关于{topic}的判断缺少直接文献支撑，只能作为工作假设提出，尚需后续材料检验。",
    "由于检索不到与{topic}直接相关的文献，本节的论述以概念推演为主，证据强度有限。",
    "在没有可比研究的情况下，{topic}的相关结论只能视为一种解释，而非定论。",
    "本节关于{topic}的观点更多来自逻辑推演，若要转为经验结论，还需补充一手材料。",
    "上述关于{topic}的判断目前只能给出方向，具体量级有待后续研究补充。",
]


def _role_index(role: str) -> int:
    return _ROLE_ORDER.index(role) if role in _ROLE_ORDER else len(_ROLE_ORDER) - 1


def build_section(prompt: str) -> str:
    """按论文题目 + 章节要点生成正文；有文献编号时按编号引用，没有就不编。"""
    info = _extract_section_prompt(prompt)
    section = info["section"]
    points = info["points"]
    citations = info["citations"]
    budget = max(200, info["word_count"])

    # 题目缺失时退回章节名：宁可用章节名做槽位，也不要生成一段与题目无关的通用文字
    title = info["paper_title"] or section
    slots = title_slots(title)
    domain = detect_domain(title, info["paper_title"])
    seed_key = _seed(title, section)

    role = _role_for(section)
    # 同一篇论文里每个角色只出现一次，所以「按角色选句 + 按题目定变体」不会撞句
    variant = _seed(slots["topic"]) % 2
    opener = _ROLE_OPENERS[role][variant].format(title=title, section=section, **slots)
    paras = [opener]

    for i, point in enumerate(points[:5]):
        lead = _LEADINS[i % len(_LEADINS)].format(point=point)
        paras.append(f"{lead}，{_evidence(domain, seed_key, i)}")

    cite_idx = _role_index(role) % len(_CITE_LINES)
    if citations:
        picks = sorted(random.Random(seed_key).sample(citations, min(2, len(citations))))
        refs = "、".join(f"[{n}]" for n in picks)
        paras.append(_CITE_LINES[cite_idx].format(refs=refs, topic=slots["topic"]))
    else:
        paras.append(_NO_CITE_LINES[cite_idx].format(topic=slots["topic"]))

    paras.append(_ROLE_CLOSINGS[role].format(section=section, title=title, **slots))

    # 篇幅补足：按大纲分配的预估字数往上贴。
    # 只补不重复的论据句，不复制已有段落——靠复制灌水会让字数达标而可信度归零。
    # 双句段落约 120 字，因此目标段数按 budget/130 折算。
    target_paras = max(4, min(15, budget // 120))
    k = 0
    while len(paras) < target_paras and k < 30:
        candidate = _evidence(domain, seed_key, k + 10)
        k += 1
        if candidate in paras:
            continue
        if len("\n\n".join(paras)) > budget * 1.15:
            break
        paras.insert(len(paras) - 1, candidate)

    return "\n\n".join(paras)


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


def _parse_review_input(prompt: str) -> dict:
    """从评审 Prompt 中取（标题, 正文, 章标题列表, 是否有引用）。"""
    title_m = re.search(r"##\s*论文标题\n(.+)", prompt)
    title = title_m.group(1).strip() if title_m else "未命名论文"
    body_m = re.search(r"##\s*正文内容\n(.*?)\n\n##\s*参考文献", prompt, re.S)
    body = body_m.group(1) if body_m else prompt
    headings = [h.strip() for h in re.findall(r"(?m)^##\s+(.+)$", body)]
    cites = re.findall(r"\[\d+\]", body)
    return {"title": title, "body": body, "headings": headings, "cites": cites}


def build_review(prompt: str) -> dict:
    """返回符合 ReviewResult Schema 的 dict。

    分数**按内容特征算**，不是按标题哈希查表：早期版本所有论文都在 0.62-0.70，
    分数与正文无关——那样演示出来的「质量门禁」等于没有。现在：
      长度影响完整度、章数影响结构、引用命中影响引用维度、
      句长分布影响表达维度。同一篇论文重复评审仍然完全一致（确定性）。
    """
    info = _parse_review_input(prompt)
    title, body, headings, cites = info["title"], info["body"], info["headings"], info["cites"]

    chars = len(body.strip())
    n_cites = len(cites)
    sentences = [s for s in re.split(r"[。！？；]", body) if s.strip()]
    avg_len = chars / max(len(sentences), 1)

    # 创新性：题目越具体（研究对象越长）说明问题越聚焦；有文献支撑再加一档
    specificity = min(1.0, len(title_slots(title)["topic"]) / 10)
    innovation = 0.48 + 0.12 * specificity + (0.10 if n_cites else 0.0)
    # 结构：章节数（6 章是学术论文的常见结构）
    structure = 0.40 + 0.055 * min(len(headings), 6)
    # 完整度：正文字数，5000 字视为内容充分（3000 字就封顶会让长文短文的分数一样）
    completeness = 0.26 + 0.58 * min(1.0, chars / 5000)
    # 表达：平均句长落在 35-45 字最自然，过长过短都扣
    language = 0.52 + 0.28 * max(0.0, 1 - abs(avg_len - 38) / 45)
    # 引用：没有引用必须低分（引用铁律），有引用按密度给分
    citation = 0.16 if not n_cites else 0.34 + 0.46 * min(1.0, n_cites / 6)

    scores = [
        round(min(0.92, innovation), 2),
        round(min(0.92, structure), 2),
        round(min(0.92, completeness), 2),
        round(min(0.92, language), 2),
        round(min(0.92, citation), 2),
    ]

    comments = [
        "选题与既有工作的差异点交代得还不够尖锐，创新性主张需要更明确的对照。",
        (
            f"章节之间的过渡基本成立，但论证链条在中段出现了断点。"
            if len(headings) >= 4 else
            "章节数量偏少，全文缺少必要的展开层次，论证推进显得仓促。"
        ),
        (
            "核心论点有支撑，部分小节缺少数据或案例的进一步展开。"
            if chars >= 2000 else
            f"正文仅 {chars} 字，多处论点停留在提出阶段，缺少展开与佐证。"
        ),
        (
            "术语使用基本统一，个别长句可以拆分以提升可读性。"
            if 25 <= avg_len <= 50 else
            "句子长度分布不均衡，长句占比偏高，影响可读性。"
        ),
        (
            f"引用编号与文末列表一致，全文使用 {n_cites} 处引用，密度在不同章节间可进一步均衡。"
            if n_cites else
            "正文未出现任何文献编号，所有断言都缺少来源，是最需要修补的一项。"
        ),
    ]

    dims = [
        {"dimension": name, "score": score, "comment": comment, "evidence": ""}
        for name, score, comment in zip(_DIMENSIONS, scores, comments)
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

    # 全文零引用视为阻断级问题：这是唯一会触发重写循环的情况。
    # 演示论文本身带引用，所以正常流程一遍过；用户建空库论文时才会看到回改。
    must_fix = [] if n_cites or chars < 120 else ["正文未使用任何参考文献编号，结论缺少证据支撑"]

    return {
        "score": round(sum(scores) / len(scores), 2),
        "dimensions": dims,
        "section_feedback": section_feedback,
        "must_fix": must_fix,
        "overall": (
            "文章结构完整、论述方向清晰，具备进一步打磨的基础；"
            "主要问题集中在创新点的表述强度与中段论证的支撑材料上。"
            if chars >= 1200 else
            "文章已具备基本框架，但目前更像提纲式展开：论点提出得多、论证展开得少，"
            "建议优先补充材料与数据，再考虑创新表述。"
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

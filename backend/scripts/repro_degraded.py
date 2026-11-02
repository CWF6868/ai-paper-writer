#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""降级复现：把输出 token 上限降回 DeepSeek 默认值，验证「评审 JSON 被截断 →
三级降级全失败 → quality_degraded=True → 落库 → API 下发」整条链路。

背景：2026-11-02 评测抓到的降级主因就是 LLM_MAX_TOKENS 未设置（落到 4096），
评审要输出 5 个维度 + 逐章建议，JSON 被截断成非法串，三级解析全失败。现在
修成了 8192，但「修好了」必须在被人为打回 4096 后仍能正确降级，而不是崩掉
或编一个假分数。

本脚本只跑「评审节点 + 落库 + 详情接口」，不跑完整工作流（不碰 RAG/Embedding），
因此不会因为 DashScope 额度问题干扰结论。

用法：
    python scripts/repro_degraded.py              # 依次尝试 4096 → 2048 → 1024
    python scripts/repro_degraded.py --max-tokens 4096
    python scripts/repro_degraded.py --keep-db    # 保留上次数据库（否则每次重建）

产物：backend/data/repro_degraded.db（可用它单独起后端看前端提示块）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DB_PATH = BASE / "data" / "repro_degraded.db"

# 必须在 import app.* 之前设置：pydantic-settings 的真实环境变量优先于 .env，
# 因此这次复现绝不碰开发库 data/paper_writer.db
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH.as_posix()}"
os.environ.setdefault("LOG_LEVEL", "INFO")

sys.path.insert(0, str(BASE))

from httpx import ASGITransport, AsyncClient          # noqa: E402
from sqlalchemy import select, text                  # noqa: E402

# 注意顺序：先 import 模型（会把名字 app 重绑为包），最后再拿 FastAPI 实例
import app.models.user      # noqa: F401,E402
import app.models.paper     # noqa: F401,E402
import app.models.chapter   # noqa: F401,E402
import app.models.reference  # noqa: F401,E402

from app.agents.reviewer_agent import ReviewerAgent   # noqa: E402
from app.api.papers import _save_results              # noqa: E402
from app.database import AsyncSessionLocal, Base, engine  # noqa: E402
from app.llm.tools.context import set_current_references  # noqa: E402
from app.main import _ensure_quality_columns, app as fastapi_app  # noqa: E402
from app.models.paper import Paper                    # noqa: E402
from app.utils.llm import get_llm                     # noqa: E402

REFERENCES = (
    "[1] Lewis P, Perez E, Piktus A, et al. Retrieval-Augmented Generation for "
    "Knowledge-Intensive NLP Tasks[C]. NeurIPS, 2020.\n"
    "[2] Ji Z, Lee N, Frieske R, et al. Survey of Hallucination in Natural Language "
    "Generation[J]. ACM Computing Surveys, 2023, 55(12): 1-38.\n"
    "[3] Gao Y, Xiong Y, Gao X, et al. Retrieval-Augmented Generation for Large "
    "Language Models: A Survey[J]. arXiv:2312.10997, 2023."
)

REF_TEXTS = [
    {"title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
     "text": "检索增强生成将参数化记忆与非参数化检索结合，用稠密向量检索相关文档"
             "作为生成条件，在知识密集型任务上降低事实性幻觉。"},
    {"title": "Survey of Hallucination in Natural Language Generation",
     "text": "幻觉指生成看似流畅但与事实不符的内容，可分为内在幻觉与外在幻觉。"},
    {"title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
     "text": "RAG 分为检索、增强、生成三阶段，检索质量与上下文噪声是关键影响因素。"},
]

# 7 章正文：章节越多，section_feedback 越长，越接近真实评审的输出规模
PAPER = """# 基于证据门控的检索增强生成幻觉缓解方法

## 摘要
大语言模型在知识密集型任务中常产生事实性幻觉。本文提出带证据门控的两阶段检索增强框架（EG-RAG），在检索后引入相关性重排序与上下文压缩，仅当证据置信度超过阈值时才注入提示。在 TruthfulQA 与 HotpotQA 上的实验表明，方法将事实准确率提升 12.3 与 8.7 个百分点，无支撑陈述比例由 18.4% 降至 7.1%，推理延迟增加 14%。

## 1 引言
幻觉是限制大语言模型落地的主要障碍[2]。引入外部知识检索可缓解该问题[1][3]。但简单的检索即拼接会引入大量噪声文档，反而干扰生成[3]。本文关注如何在高召回与低噪声之间加入一道证据质量门控。

## 2 相关工作
Lewis 等提出 RAG，将稠密检索与生成模型联合训练[1]。Ji 等梳理了幻觉的分类与成因[2]。Gao 等的综述指出检索噪声是影响 RAG 效果的关键变量[3]。上述工作均未对注入证据做逐句置信度校验。

## 3 问题定义
给定查询 q 与候选文档集 D，目标是选出一个子集 S，使得在 |S| 受限的前提下，生成结果的事实一致性最大化。本文将该问题建模为带预算约束的子集选择问题。

## 4 方法
框架包含三部分。稠密检索用双塔编码器召回 top-20 文档；相关性重排序用交叉编码器打分，保留分数高于 0.62 的段落；证据门控对每个待生成论断计算语义蕴含分数，仅当超过 0.55 时注入。阈值由验证集网格搜索确定。

## 5 实验设置
数据集为 TruthfulQA 与 HotpotQA，基线为原始 RAG[1] 与无检索的 LLaMA-2-7B。评价指标为事实准确率与无支撑陈述比例。所有实验在单卡 A100 上完成，随机种子取 3 次平均。

## 6 结果与分析
EG-RAG 在 TruthfulQA 上事实准确率 61.2%，较基线 48.9% 提升 12.3 个百分点；无支撑陈述比例由 18.4% 降至 7.1%。消融显示去掉证据门控后提升收窄至 5.1 个百分点，说明门控是关键组件。延迟从 1.42s 增至 1.62s。

## 7 结论
本文通过证据质量门控抑制了检索噪声导致的幻觉。局限在于阈值需按领域重新标定，跨领域迁移性有待验证。
"""


def _reset_db() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()


async def _review_once(max_tokens: int) -> dict:
    """真实调用一次评审，返回 {raised, mode, score, error}"""
    agent = ReviewerAgent(llm=get_llm(temperature=0.0, max_tokens=max_tokens))
    try:
        with set_current_references(REF_TEXTS):
            result = await agent.execute({
                "title": "基于证据门控的检索增强生成幻觉缓解方法",
                "topic": "大语言模型幻觉缓解与检索增强生成",
                "content": PAPER,
                "references": REFERENCES,
            })
        return {
            "raised": None,
            "mode": "function_calling" if not result.get("degraded") else "降级后成功",
            "score": (result.get("data") or {}).get("score"),
            "dims": len((result.get("data") or {}).get("dimensions") or []),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 —— 复现的就是这条异常路径
        return {
            "raised": type(e).__name__,
            "mode": None,
            "score": None,
            "dims": 0,
            "error": str(e),
        }


async def _persist_and_verify(reason: str) -> dict:
    """模拟 graph_builder._degraded 的产物 → 真实落库 → 真实详情接口"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_quality_columns(conn)

    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://repro") as client:
        reg = await client.post("/api/auth/register", json={
            "username": "repro_user", "email": "repro@example.com", "password": "repro123456",
        })
        if reg.status_code not in (201, 400):     # 400 = 上次跑过，账号已存在
            print(f"  注册异常：{reg.status_code} {reg.text[:200]}")
        # 登录是 OAuth2PasswordRequestForm（表单），不是 JSON
        login = await client.post("/api/auth/login", data={
            "username": "repro@example.com", "password": "repro123456",
        })
        if login.status_code != 200:
            raise RuntimeError(f"登录失败：{login.status_code} {login.text[:200]}")
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        created = await client.post("/api/papers/", json={
            "title": "降级复现：证据门控的检索增强生成幻觉缓解",
            "topic": "大语言模型幻觉缓解",
        }, headers=headers)
        pid = created.json()["id"]

        # 与 graph_builder._quality_check 的 _degraded() 返回结构保持一致
        state = {
            "final_paper": PAPER,
            "outline": {"sections": []},
            "sections_content": {"摘要": "（略）"},
            "references": [],
            "quality_score": None,
            "quality_degraded": True,
            "quality_comment": f"评审不可用（{reason}），本轮跳过质量门禁",
            "quality_dimensions": [],
        }
        async with AsyncSessionLocal() as db:
            paper = (await db.execute(select(Paper).where(Paper.id == pid))).scalar_one()
            await _save_results(paper, state, db)
            await db.commit()

        detail = (await client.get(f"/api/papers/{pid}", headers=headers)).json()
        async with AsyncSessionLocal() as db:
            row = (await db.execute(
                text("SELECT quality_score, quality_degraded FROM papers WHERE id = :i"),
                {"i": pid},
            )).first()

    return {
        "paper_id": pid,
        "db_quality_score": row[0],
        "db_quality_degraded": row[1],
        "api_quality_score": detail.get("quality_score"),
        "api_quality_degraded": detail.get("quality_degraded"),
        "api_comment": detail.get("quality_comment"),
        "content_len": len(detail.get("content") or ""),
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description="评审降级端到端复现")
    ap.add_argument("--max-tokens", type=int, default=None, help="固定尝试某个上限")
    ap.add_argument("--keep-db", action="store_true", help="保留上次的复现库")
    args = ap.parse_args()

    if not args.keep_db:
        _reset_db()

    candidates = [args.max_tokens] if args.max_tokens else [4096, 2048, 1024]
    print("=" * 72)
    print("步骤 1/3：真实评审调用（DeepSeek），逐级观察三级降级")
    print("=" * 72)

    trigger = None
    for mt in candidates:
        print(f"\n--- max_tokens = {mt} ---", flush=True)
        r = await _review_once(mt)
        if r["raised"]:
            print(f"  评审失败：{r['raised']}")
            print(f"  最后错误：{(r['error'] or '')[:400]}")
            trigger = {"max_tokens": mt, **r}
            break
        print(f"  评审成功：score={r['score']}，维度数={r['dims']}（{mt} 未触发截断）")

    if trigger is None:
        print("\n结论：所有候选上限都没触发截断 —— 当前输出规模下不会复现，")
        print("      可把 --max-tokens 再调低（如 512）重试。")
        await engine.dispose()
        return 1

    print()
    print("=" * 72)
    print("步骤 2/3：按 graph_builder._degraded() 的产物落库")
    print("=" * 72)
    reason = f"{trigger['raised']}: {(trigger['error'] or '')[:80]}"
    info = await _persist_and_verify(reason)
    print(f"  paper_id            = {info['paper_id']}")
    print(f"  DB.quality_score    = {info['db_quality_score']}")
    print(f"  DB.quality_degraded = {info['db_quality_degraded']}")

    print()
    print("=" * 72)
    print("步骤 3/3：详情接口实际返回（前端渲染提示块的数据源）")
    print("=" * 72)
    print(json.dumps({
        "quality_score": info["api_quality_score"],
        "quality_degraded": info["api_quality_degraded"],
        "quality_comment": info["api_comment"],
        "content_len": info["content_len"],
    }, ensure_ascii=False, indent=2))

    ok = (
        info["db_quality_degraded"] in (1, True)
        and info["api_quality_degraded"] is True
        and info["api_quality_score"] is None
    )
    print()
    print(("✅ 链路正确：截断 → 三级降级全失败 → 显式降级 → 落库 → API 下发 True"
           if ok else "❌ 链路异常：请检查上面的落库/接口结果"))
    print(f"复现库：{DB_PATH}")
    print("看前端提示块：把 .env 的 DATABASE_URL 临时指向该库后起后端 + npm run dev，")
    print("              登录 repro@example.com / repro123456 打开这篇论文。")

    await engine.dispose()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

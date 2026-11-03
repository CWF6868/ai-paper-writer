# smoke_demo.py —— 演示实例冒烟脚本（本地 / 公网同一份）
#
# 验证的是一条完整用户路径，而不是"接口返回 200"：
#   健康检查 → 登录 → 建论文 → SSE 生成 → 详情（章节/文献/评分）→ 导出 docx/pdf
# 任何一步不满足预期都会以非 0 退出，方便在 CI / 部署后直接当门禁跑。
#
# 用法：
#   python scripts/smoke_demo.py                                   # 本地 8000
#   python scripts/smoke_demo.py --base https://xxx.onrender.com   # 公网
#   python scripts/smoke_demo.py --public                          # 公网时带上"演示模式"断言
import argparse
import json
import sys
import time

import httpx

DEFAULT_EMAIL = "demo@example.com"
DEFAULT_PASSWORD = "demo-readonly"


def _fail(step: str, detail: str) -> None:
    print(f"[FAIL] {step}: {detail}")
    sys.exit(1)


def _ok(step: str, detail: str = "") -> None:
    print(f"[ OK ] {step}{(' — ' + detail) if detail else ''}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--public", action="store_true", help="公网实例：额外校验首页与演示模式标识")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    with httpx.Client(timeout=180, follow_redirects=True) as c:
        # 1) 健康检查
        try:
            r = c.get(f"{base}/")
        except Exception as exc:  # noqa: BLE001
            _fail("健康检查", f"无法连接 {base}: {exc}")
        if r.status_code != 200:
            _fail("健康检查", f"HTTP {r.status_code}")
        _ok("健康检查", r.json().get("status", ""))

        # 2) 登录（演示账号）
        r = c.post(f"{base}/api/auth/login", data={"username": args.email, "password": args.password})
        if r.status_code != 200:
            _fail("登录", f"HTTP {r.status_code} {r.text[:120]}")
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        _ok("登录", args.email)

        # 3) 建论文（标题带时间戳，避免重复运行撞名）
        title = f"演示冒烟验证 {time.strftime('%Y-%m-%d %H:%M:%S')}"
        r = c.post(f"{base}/api/papers/", headers=headers, json={
            "title": title, "topic": "冒烟验证", "keywords": "验证,自动化",
        })
        if r.status_code != 201:
            _fail("建论文", f"HTTP {r.status_code} {r.text[:120]}")
        paper_id = r.json()["id"]
        _ok("建论文", f"id={paper_id}")

        # 4) SSE 生成：至少要看到节点事件和 done 事件
        t0 = time.time()
        events, done, error = 0, None, None
        with c.stream("POST", f"{base}/api/papers/{paper_id}/stream-generate", headers=headers) as s:
            if s.status_code != 200:
                _fail("SSE 生成", f"HTTP {s.status_code}")
            for line in s.iter_lines():
                if not line.startswith("data: "):
                    continue
                events += 1
                payload = json.loads(line[6:])
                if payload.get("done"):
                    done = payload
                elif payload.get("error"):
                    error = payload["error"]
        elapsed = time.time() - t0
        if error:
            _fail("SSE 生成", error[:200])
        if not done:
            _fail("SSE 生成", f"没有收到 done 事件（收到 {events} 条）")
        score = done.get("quality_score")
        if score is None:
            _fail("评审", f"quality_score 为空（degraded={done.get('quality_degraded')}）")
        if not (0.0 <= float(score) <= 1.0):
            _fail("评审", f"分数越界 {score}")
        _ok("SSE 生成", f"{events} 条事件 / {elapsed:.1f}s / 评分 {score}")

        # 5) 详情：章节、文献、正文都要真的落库
        r = c.get(f"{base}/api/papers/{paper_id}", headers=headers)
        if r.status_code != 200:
            _fail("论文详情", f"HTTP {r.status_code}")
        detail = r.json()
        chapters = detail.get("chapters") or []
        refs = detail.get("references") or []
        content = detail.get("content") or ""
        if len(chapters) < 3:
            _fail("论文详情", f"章节数异常：{len(chapters)}")
        if len(content) < 500:
            _fail("论文详情", f"正文过短：{len(content)} 字")
        _ok("论文详情", f"{len(chapters)} 章 / {len(refs)} 篇文献 / 正文 {len(content)} 字")

        # 6) 导出：只看 200 不够，要校验文件头，避免"500 被包装成 200"
        for fmt, magic in (("docx", b"PK\x03\x04"), ("pdf", b"%PDF")):
            r = c.get(f"{base}/api/papers/{paper_id}/export", params={"format": fmt}, headers=headers)
            if r.status_code != 200:
                _fail(f"导出 {fmt}", f"HTTP {r.status_code} {r.text[:120]}")
            if not r.content.startswith(magic):
                _fail(f"导出 {fmt}", f"文件头不是 {magic!r}，长度 {len(r.content)}")
            if len(r.content) < 2000:
                _fail(f"导出 {fmt}", f"文件过小 {len(r.content)} 字节，疑似空文档")
            _ok(f"导出 {fmt}", f"{len(r.content) / 1024:.1f} KB")

        # 7) 公网额外检查：根路径可达（前端由 nginx 托管时这里应是 HTML）
        if args.public:
            r = c.get(f"{base}/")
            if "text/html" not in r.headers.get("content-type", "") and r.headers.get("content-type", "").startswith("application/json"):
                print("[WARN] 根路径返回 JSON：前端未由同一域名托管（仅 API 可访问）")

    print("\n冒烟通过：登录 → 生成 → 评审 → 落库 → 导出 全链路可用")


if __name__ == "__main__":
    main()

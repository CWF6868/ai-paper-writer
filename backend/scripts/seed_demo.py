# seed_demo.py —— 手动初始化演示数据（幂等）
#
# 用法：
#   DEMO_MODE=true python scripts/seed_demo.py
#   python scripts/seed_demo.py            # 自动把 DEMO_MODE 置为 true
#
# 容器里由启动脚本自动调用，本地一般用不到；排查"Demo 首屏为什么是空的"时手动跑一次。
import asyncio
import os
import sys

# 必须先置环境变量再 import app：配置在模块导入时就冻结了
os.environ.setdefault("DEMO_MODE", "true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.demo.seed import ensure_demo_data  # noqa: E402


async def main():
    stats = await ensure_demo_data()
    print("演示数据就绪：", stats)
    from app.demo.seed import DEMO_EMAIL, DEMO_PASSWORD
    print(f"演示账号：{DEMO_EMAIL} / {DEMO_PASSWORD}")


if __name__ == "__main__":
    asyncio.run(main())

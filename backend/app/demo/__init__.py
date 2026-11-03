# demo/__init__.py —— 演示模式（公开 Demo 用）
#
# 只有一个开关 settings.DEMO_MODE：打开后模型调用与向量化都换成本地确定性实现，
# 其余代码路径（工作流 / 质量门禁 / SSE / 导出）保持不变。
from app.demo.chat import DemoChatModel, DemoHashEmbeddings  # noqa: F401

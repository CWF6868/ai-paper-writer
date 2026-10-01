from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from app.database import Base

def utc_now():
    """返回当前 UTC 时间（datetime.utcnow 的新式写法）"""
    return datetime.now(timezone.utc)

class Chapter(Base):
    """章节表：论文的每个章节"""
    __tablename__ = "chapters"

    id = Column(Integer, primary_key=True, index=True)              # 主键
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)   # 属于哪篇论文（必填）
    parent_id = Column(Integer, ForeignKey("chapters.id"), nullable=True)  # 父章节（支持子章节，可为空）
    order_index = Column(Integer, default=0)                        # 排序序号，决定章节先后

    title = Column(String(200), nullable=False)                     # 章节标题（必填）
    content = Column(Text)                                          # 章节正文内容
    summary = Column(Text)                                          # 章节摘要

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # 关系：属于某篇论文；子章节（children 指向 parent）
    paper = relationship("Paper", back_populates="chapters")
    children = relationship("Chapter", backref="parent", remote_side=[id])

    def __repr__(self):
        return f"<Chapter(id={self.id}, title='{self.title}')>"
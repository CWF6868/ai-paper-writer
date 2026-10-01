import enum

from sqlalchemy import Column, Integer, Float, String, Text, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from app.database import Base

def utc_now():
    """返回当前 UTC 时间（datetime.utcnow 的新式写法）"""
    return datetime.now(timezone.utc)

class PaperStatus(enum.Enum):
    """论文状态的枚举（草稿→大纲→撰写→审核→完成）"""
    DRAFT = "draft"
    OUTLINE = "outline"
    WRITING = "writing"
    REVIEW = "review"
    COMPLETED = "completed"

class Paper(Base):
    """论文表：一篇论文的所有元信息"""
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)              # 主键
    title = Column(String(500), nullable=False)                     # 论文标题（必填）
    topic = Column(String(200))                                     # 选题方向
    keywords = Column(String(500))                                  # 关键词，逗号分隔
    abstract = Column(Text)                                         # 摘要
    status = Column(Enum(PaperStatus), default=PaperStatus.DRAFT)   # 当前状态
    outline = Column(Text)                                          # 大纲（JSON 文本）
    content = Column(Text)                                          # AI 生成的论文全文
    quality_score = Column(Float)                                   # 质量评审综合分（0~1）
    quality_comment = Column(Text)                                  # 评审意见
    quality_dimensions = Column(Text)                               # 各维度评分明细（JSON 文本）

    author_id = Column(Integer, ForeignKey("users.id"))             # 关联作者
    created_at = Column(DateTime, default=utc_now)                  # 创建时间
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)  # 更新时间，修改时自动刷新

    # 关系：作者、章节（删除论文时级联删除章节）、文献
    author = relationship("User", back_populates="papers")
    chapters = relationship("Chapter", back_populates="paper", cascade="all, delete-orphan")
    references = relationship("Reference", back_populates="paper", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Paper(id={self.id}, title='{self.title}', status='{self.status}')>"
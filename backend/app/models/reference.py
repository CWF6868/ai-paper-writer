from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship

from app.database import Base

class Reference(Base):
    """参考文献表：论文引用的文献元信息"""
    __tablename__ = "references"

    id = Column(Integer, primary_key=True, index=True)          # 主键
    paper_id = Column(Integer, ForeignKey("papers.id"))         # 属于哪篇论文（可为空）

    title = Column(String(500), nullable=False)                 # 文献标题（必填）
    authors = Column(String(500))                               # 作者列表
    journal = Column(String(200))                               # 期刊名
    year = Column(Integer)                                      # 发表年份
    volume = Column(String(50))                                 # 卷号
    issue = Column(String(50))                                  # 期号
    pages = Column(String(50))                                  # 页码
    doi = Column(String(100))                                   # DOI 唯一标识
    url = Column(String(500))                                   # 链接
    abstract = Column(Text)                                     # 摘要

    vector_id = Column(String(100))                             # 在向量库里的编号（供 RAG 检索用）

    paper = relationship("Paper", back_populates="references")  # 关联论文

    def __repr__(self):
        return f"<Reference(id={self.id}, title='{self.title}')>"
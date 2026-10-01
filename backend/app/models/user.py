from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from app.database import Base

def utc_now():
    """返回当前 UTC 时间（datetime.utcnow 的新式写法，避免弃用告警）"""
    return datetime.now(timezone.utc)

class User(Base):
    """用户表：记录系统登录账号"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)              # 主键，自增
    username = Column(String(50), unique=True, nullable=False, index=True)   # 用户名，唯一
    email = Column(String(100), unique=True, nullable=False, index=True)     # 邮箱，唯一
    hashed_password = Column(String(200), nullable=False)                    # 加密后的密码（bcrypt）

    is_active = Column(Boolean, default=True)        # 账号是否启用
    is_superuser = Column(Boolean, default=False)    # 是否为管理员

    created_at = Column(DateTime, default=utc_now)   # 注册时间
    last_login = Column(DateTime, nullable=True)     # 最近登录时间

    # 关系：一个用户拥有多篇论文（一对多）
    papers = relationship("Paper", back_populates="author")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"
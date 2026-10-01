# user.py —— 用户相关的 Pydantic 模式（API 进出口校验）
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from datetime import datetime


class UserCreate(BaseModel):
    """新建用户（注册）时前端必须传来的字段"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱")
    password: str = Field(..., min_length=6, max_length=128, description="密码")


class UserLogin(BaseModel):
    """登录时前端必须传来的字段"""
    username: str = Field(..., description="用户名或邮箱")
    password: str = Field(..., description="密码")


class UserOut(BaseModel):
    """返回给前端的用户信息（绝不含密码）"""
    id: int
    username: str
    email: EmailStr
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)  # 允许直接从数据库模型对象转换


class Token(BaseModel):
    """登录成功返回的 JWT 令牌"""
    access_token: str
    token_type: str = "bearer"
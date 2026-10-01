# api/auth.py —— 用户注册 / 登录 / 当前用户信息
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, verify_password, create_access_token
from app.core.deps import get_current_user
from app.database import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserOut, Token

router = APIRouter()

@router.post("/register", response_model=UserOut, status_code=201)
async def register(data: UserCreate, db: AsyncSession = Depends(get_db)):
    """注册：用户名、邮箱都不能重复"""
    exists = (await db.execute(
        select(User).where(or_(User.username == data.username, User.email == data.email))
    )).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=400, detail="用户名或邮箱已被注册")

    user = User(
        username=data.username,
        email=data.email,
        hashed_password=hash_password(data.password),  # 只存哈希
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

@router.post("/login", response_model=Token)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """
    登录：用户名或邮箱 + 密码（表单格式，Swagger Authorize 按钮直接可用）。
    成功返回 JWT。
    """
    result = await db.execute(
        select(User).where(or_(User.username == form.username, User.email == form.username))
    )
    user = result.scalar_one_or_none()

    # 账号不存在和密码错误用同一提示：防止攻击者批量试探哪些账号已注册
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="账号已被禁用")

    user.last_login = datetime.now(timezone.utc)
    await db.commit()

    return Token(access_token=create_access_token(str(user.id)))

@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)):
    """返回当前登录用户（也用来验证 token 是否有效）"""
    return current_user
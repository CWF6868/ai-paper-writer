# core/deps.py —— 从请求头的 JWT 解析出当前用户（接口守卫）
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.database import get_db
from app.models.user import User

# tokenUrl 指向登录接口：Swagger 的 Authorize 按钮靠它找到登录入口
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    守卫：解析 Authorization: Bearer <token>，查出当前用户。
    任何受保护接口只要加 Depends(get_current_user) 即可上锁。
    """
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="登录凭证无效或已过期",
        headers={"WWW-Authenticate": "Bearer"},
    )

    sub = decode_access_token(token)
    if sub is None:
        raise credentials_error
    try:
        user_id = int(sub)
    except ValueError:
        raise credentials_error

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_error
    if not user.is_active:
        raise HTTPException(status_code=403, detail="账号已被禁用")
    return user
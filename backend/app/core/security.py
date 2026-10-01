# core/security.py —— 密码哈希 + JWT 令牌
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from passlib.context import CryptContext

from app.config import settings

# 密码哈希上下文：bcrypt 算法，每次哈希自动加随机盐
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """把明文密码变成不可逆的哈希值（存库用，绝不存明文）"""
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    """校验：用户输入的明文和库里的哈希是否匹配"""
    return pwd_context.verify(plain, hashed)

def create_access_token(subject: str, expires_minutes: Optional[int] = None) -> str:
    """
    签发 JWT。
    subject 放用户 id（字符串形式）；payload 带过期时间，
    过期后 jwt.decode 会拒绝，实现"30 分钟掉线"。
    """
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def decode_access_token(token: str) -> Optional[str]:
    """
    解码 JWT，返回 subject（用户 id）。
    签名被篡改 / 已过期 / 格式错 → 返回 None（由上层统一 401）。
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload.get("sub")
    except jwt.PyJWTError:
        return None
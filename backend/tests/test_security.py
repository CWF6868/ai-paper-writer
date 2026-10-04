# 单元测试：core/security.py（密码哈希 + JWT 签发/解码）
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_is_not_plaintext_and_verifiable():
    hashed = hash_password("secret123")
    assert hashed != "secret123"           # 绝不存明文
    assert hashed.startswith("$2")         # bcrypt 哈希前缀
    assert verify_password("secret123", hashed)


def test_hash_is_salted():
    """同一密码两次哈希结果不同（随机盐生效）"""
    assert hash_password("same-pass") != hash_password("same-pass")


def test_verify_password_rejects_wrong_password():
    hashed = hash_password("secret123")
    assert not verify_password("wrong-pass", hashed)


def test_create_and_decode_token_roundtrip():
    token = create_access_token("42")
    assert decode_access_token(token) == "42"


def test_decode_token_rejects_garbage():
    assert decode_access_token("not.a.token") is None


def test_decode_token_rejects_expired():
    """过期令牌必须解不出来（实现 30 分钟掉线）"""
    token = create_access_token("7", expires_minutes=-1)
    assert decode_access_token(token) is None


def test_decode_token_rejects_tampered_signature():
    token = create_access_token("7")
    head, payload, sig = token.split(".")
    tampered = ".".join([head, payload, sig[:-2] + "xx"])
    assert decode_access_token(tampered) is None


def test_decode_token_rejects_wrong_secret():
    import jwt

    forged = jwt.encode({"sub": "1"}, "another-secret", algorithm="HS256")
    assert decode_access_token(forged) is None

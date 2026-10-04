# 集成测试：认证接口（注册 / 登录 / me）+ 鉴权守卫的各条失败路径
from helpers import register_and_login

from app.core.security import create_access_token
from app.database import AsyncSessionLocal
from app.models.user import User
from sqlalchemy import select


# ---------- 注册 ----------

async def test_register_ok(client):
    r = await client.post("/api/auth/register", json={
        "username": "alice", "email": "alice@test.com", "password": "secret123",
    })
    assert r.status_code == 201
    body = r.json()
    assert body["username"] == "alice"
    assert body["email"] == "alice@test.com"
    assert body["is_active"] is True
    assert "password" not in body and "hashed_password" not in body


async def test_register_stores_hashed_password(client):
    """库里绝不能出现明文密码"""
    await client.post("/api/auth/register", json={
        "username": "hashcheck", "email": "hashcheck@test.com", "password": "plain-secret",
    })
    async with AsyncSessionLocal() as db:
        user = (await db.execute(select(User).where(User.username == "hashcheck"))).scalar_one()
    assert user.hashed_password != "plain-secret"
    assert user.hashed_password.startswith("$2")


async def test_register_duplicate_username(client):
    payload = {"username": "bob", "email": "bob@test.com", "password": "secret123"}
    assert (await client.post("/api/auth/register", json=payload)).status_code == 201
    r = await client.post("/api/auth/register", json=payload)
    assert r.status_code == 400


async def test_register_duplicate_email(client):
    p1 = {"username": "bob", "email": "same@test.com", "password": "secret123"}
    p2 = {"username": "bob2", "email": "same@test.com", "password": "secret123"}
    assert (await client.post("/api/auth/register", json=p1)).status_code == 201
    assert (await client.post("/api/auth/register", json=p2)).status_code == 400


async def test_register_rejects_short_username(client):
    r = await client.post("/api/auth/register", json={
        "username": "ab", "email": "ab@test.com", "password": "secret123",
    })
    assert r.status_code == 422


async def test_register_rejects_bad_email(client):
    r = await client.post("/api/auth/register", json={
        "username": "carol", "email": "not-an-email", "password": "secret123",
    })
    assert r.status_code == 422


async def test_register_rejects_short_password(client):
    r = await client.post("/api/auth/register", json={
        "username": "carol", "email": "carol@test.com", "password": "123",
    })
    assert r.status_code == 422


# ---------- 登录 ----------

async def test_login_ok(client):
    await register_and_login(client, "carol")
    r = await client.post("/api/auth/login", data={"username": "carol", "password": "test123456"})
    assert r.status_code == 200
    assert r.json()["access_token"]
    assert r.json()["token_type"] == "bearer"


async def test_login_accepts_email_as_username(client):
    await register_and_login(client, "dave")
    r = await client.post("/api/auth/login", data={"username": "dave@test.com", "password": "test123456"})
    assert r.status_code == 200


async def test_login_wrong_password(client):
    await register_and_login(client, "dave")
    r = await client.post("/api/auth/login", data={"username": "dave", "password": "wrongpass"})
    assert r.status_code == 401


async def test_login_unknown_user(client):
    r = await client.post("/api/auth/login", data={"username": "nobody", "password": "whatever1"})
    assert r.status_code == 401


async def test_login_records_last_login(client):
    await register_and_login(client, "erin")
    async with AsyncSessionLocal() as db:
        user = (await db.execute(select(User).where(User.username == "erin"))).scalar_one()
        assert user.last_login is not None


async def test_login_inactive_account_forbidden(client):
    await register_and_login(client, "frank")
    async with AsyncSessionLocal() as db:
        user = (await db.execute(select(User).where(User.username == "frank"))).scalar_one()
        user.is_active = False
        await db.commit()
    r = await client.post("/api/auth/login", data={"username": "frank", "password": "test123456"})
    assert r.status_code == 403


# ---------- me ----------

async def test_me_with_token(client, headers):
    r = await client.get("/api/auth/me", headers=headers)
    assert r.status_code == 200
    assert r.json()["username"] == "tester"


async def test_me_without_token(client):
    assert (await client.get("/api/auth/me")).status_code == 401


async def test_me_with_garbage_token(client):
    r = await client.get("/api/auth/me", headers={"Authorization": "Bearer not.a.token"})
    assert r.status_code == 401


async def test_me_with_expired_token(client, token):
    async with AsyncSessionLocal() as db:
        uid = (await db.execute(select(User).where(User.username == "tester"))).scalar_one().id
    expired = create_access_token(str(uid), expires_minutes=-1)
    r = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401


async def test_me_with_non_numeric_subject(client):
    """sub 不是数字（token 被伪造）→ 401，而不是 500"""
    r = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {create_access_token('abc')}"})
    assert r.status_code == 401


async def test_me_with_deleted_user(client):
    """token 合法但用户已被删除 → 401"""
    token = create_access_token("999999")
    r = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


async def test_me_when_account_disabled(client, headers):
    async with AsyncSessionLocal() as db:
        user = (await db.execute(select(User).where(User.username == "tester"))).scalar_one()
        user.is_active = False
        await db.commit()
    assert (await client.get("/api/auth/me", headers=headers)).status_code == 403

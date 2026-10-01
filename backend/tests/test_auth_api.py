# 集成测试：认证接口（注册 / 登录 / me）
from helpers import register_and_login


async def test_register_ok(client):
    r = await client.post("/api/auth/register", json={
        "username": "alice", "email": "alice@test.com", "password": "secret123",
    })
    assert r.status_code == 201
    body = r.json()
    assert body["username"] == "alice"
    assert body["email"] == "alice@test.com"
    assert "password" not in body          # 绝不返回密码


async def test_register_duplicate(client):
    payload = {"username": "bob", "email": "bob@test.com", "password": "secret123"}
    assert (await client.post("/api/auth/register", json=payload)).status_code == 201
    r = await client.post("/api/auth/register", json=payload)
    assert r.status_code == 400


async def test_register_duplicate_email(client):
    p1 = {"username": "bob", "email": "same@test.com", "password": "secret123"}
    p2 = {"username": "bob2", "email": "same@test.com", "password": "secret123"}
    assert (await client.post("/api/auth/register", json=p1)).status_code == 201
    assert (await client.post("/api/auth/register", json=p2)).status_code == 400


async def test_login_ok(client):
    await register_and_login(client, "carol")
    r = await client.post("/api/auth/login", data={"username": "carol", "password": "test123456"})
    assert r.status_code == 200
    assert r.json()["access_token"]


async def test_login_wrong_password(client):
    await register_and_login(client, "dave")
    r = await client.post("/api/auth/login", data={"username": "dave", "password": "wrongpass"})
    assert r.status_code == 401


async def test_login_unknown_user(client):
    r = await client.post("/api/auth/login", data={"username": "nobody", "password": "whatever1"})
    assert r.status_code == 401


async def test_me_with_token(client, headers):
    r = await client.get("/api/auth/me", headers=headers)
    assert r.status_code == 200
    assert r.json()["username"] == "tester"


async def test_me_without_token(client):
    assert (await client.get("/api/auth/me")).status_code == 401


async def test_me_with_garbage_token(client):
    r = await client.get("/api/auth/me", headers={"Authorization": "Bearer not.a.token"})
    assert r.status_code == 401

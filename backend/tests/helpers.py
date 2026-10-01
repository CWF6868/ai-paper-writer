# 集成测试公共 helper（不以下划线/ test_ 开头，避免被 pytest 收集）
async def register_and_login(client, username="tester", password="test123456", email=None):
    """注册 + 登录，返回 access_token"""
    email = email or f"{username}@test.com"
    r = await client.post("/api/auth/register", json={
        "username": username, "email": email, "password": password,
    })
    assert r.status_code == 201, r.text
    r = await client.post("/api/auth/login", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]

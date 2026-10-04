# 单元测试：schemas（Pydantic 校验）+ models（ORM 定义）+ main 的轻量迁移
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.database import engine
from app.main import _ensure_quality_columns
from app.models.chapter import Chapter
from app.models.paper import Paper, PaperStatus, utc_now
from app.models.reference import Reference
from app.models.user import User
from app.schemas.chapter import ChapterCreate, ChapterDetail, ChapterOut
from app.schemas.paper import PaperCreate, PaperDetail, PaperUpdate
from app.schemas.topic import TopicRecommendRequest, TopicRecommendResponse
from app.schemas.user import Token, UserCreate, UserOut


# ---------- schemas/user ----------

def test_user_create_ok():
    u = UserCreate(username="abc", email="a@b.com", password="secret1")
    assert u.username == "abc"


@pytest.mark.parametrize("payload", [
    {"username": "ab", "email": "a@b.com", "password": "secret1"},        # 用户名太短
    {"username": "abc", "email": "not-an-email", "password": "secret1"},  # 邮箱非法
    {"username": "abc", "email": "a@b.com", "password": "123"},           # 密码太短
])
def test_user_create_rejects_invalid(payload):
    with pytest.raises(ValidationError):
        UserCreate(**payload)


def test_user_out_disallows_password_field():
    """UserOut 里没有密码字段，即便传入也会被忽略（防泄漏）"""
    u = UserOut(id=1, username="a", email="a@b.com", is_active=True,
                created_at=datetime.now(timezone.utc), password="leak")
    assert not hasattr(u, "password")


def test_token_defaults_to_bearer():
    assert Token(access_token="x").token_type == "bearer"


# ---------- schemas/paper ----------

def test_paper_create_requires_five_char_title():
    assert PaperCreate(title="五个字的标题").title == "五个字的标题"
    with pytest.raises(ValidationError):
        PaperCreate(title="短")


def test_paper_update_all_optional():
    assert PaperUpdate().model_dump(exclude_unset=True) == {}
    upd = PaperUpdate(abstract="新摘要")
    assert upd.model_dump(exclude_unset=True) == {"abstract": "新摘要"}


def test_paper_detail_defaults_lists_empty():
    d = PaperDetail(id=1, title="标题标题标题", status=PaperStatus.DRAFT,
                    created_at=utc_now(), updated_at=utc_now())
    assert d.chapters == [] and d.references == []
    assert d.quality_score is None


def test_paper_status_values():
    assert PaperStatus.DRAFT.value == "draft"
    assert PaperStatus.COMPLETED.value == "completed"
    assert {s.value for s in PaperStatus} == {"draft", "outline", "writing", "review", "completed"}


# ---------- schemas/topic ----------

def test_topic_request_trims_nothing_but_validates_length():
    assert TopicRecommendRequest(field="AI").field == "AI"
    with pytest.raises(ValidationError):
        TopicRecommendRequest(field="")


def test_topic_response_wraps_string():
    assert TopicRecommendResponse(topics="1. A").topics == "1. A"


# ---------- schemas/chapter ----------

def test_chapter_schemas():
    c = ChapterCreate(title="引言", paper_id=1)
    assert c.order_index == 0 and c.parent_id is None
    out = ChapterOut(id=1, paper_id=1, parent_id=None, order_index=0, title="引言",
                     created_at=utc_now(), updated_at=utc_now())
    assert ChapterDetail(**out.model_dump(), children=[]).children == []


# ---------- models ----------

def test_utc_now_is_timezone_aware():
    now = utc_now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timezone.utc.utcoffset(None)


def test_model_reprs():
    assert repr(User(id=1, username="u1")).startswith("<User(id=1")
    assert "Paper(id=2" in repr(Paper(id=2, title="T", status=PaperStatus.DRAFT))
    assert repr(Chapter(id=3, title="引言")) == "<Chapter(id=3, title='引言')>"
    assert repr(Reference(id=4, title="论文A")) == "<Reference(id=4, title='论文A')>"


def test_model_table_names():
    assert User.__tablename__ == "users"
    assert Paper.__tablename__ == "papers"
    assert Chapter.__tablename__ == "chapters"
    assert Reference.__tablename__ == "references"


def test_paper_relationships_configured():
    rels = {r.key for r in Paper.__mapper__.relationships}
    assert rels == {"author", "chapters", "references"}
    assert "papers" in {r.key for r in User.__mapper__.relationships}


# ---------- main 的轻量迁移 ----------

async def test_ensure_quality_columns_is_idempotent():
    """重复执行不报错（列已存在就跳过）"""
    async with engine.begin() as conn:
        await _ensure_quality_columns(conn)
        await _ensure_quality_columns(conn)

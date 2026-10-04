# 单元测试：utils/llm.py（模型接入层 + Key 校验）
import pytest
from fastapi import HTTPException
from langchain_openai import ChatOpenAI

from app.config import settings
from app.utils.llm import check_llm_config, get_embedding_model, get_llm


def test_get_llm_returns_deepseek_chat_model():
    llm = get_llm()
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == settings.DEEPSEEK_MODEL


def test_get_llm_respects_explicit_provider_and_model():
    llm = get_llm(provider="deepseek", model="deepseek-chat", temperature=0.2)
    assert llm.model_name == "deepseek-chat"
    assert llm.temperature == 0.2


def test_get_llm_rejects_unknown_provider():
    with pytest.raises(ValueError, match="不支持的模型提供商"):
        get_llm(provider="openai")


def test_get_embedding_model_disables_ctx_length_check():
    """百炼兼容接口只吃纯文本，必须关掉 langchain 的 token 预处理"""
    emb = get_embedding_model()
    assert emb.check_embedding_ctx_length is False
    assert emb.model == settings.DASHSCOPE_MODEL


def test_check_llm_config_passes_when_both_keys_present(monkeypatch):
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "k1")
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "k2")
    check_llm_config()                       # 不抛异常即通过


def test_check_llm_config_reports_both_missing_keys(monkeypatch):
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", None)
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", None)
    with pytest.raises(HTTPException) as ei:
        check_llm_config()
    assert ei.value.status_code == 400
    assert "DEEPSEEK_API_KEY" in ei.value.detail
    assert "DASHSCOPE_API_KEY" in ei.value.detail


def test_check_llm_config_can_skip_embedding_requirement(monkeypatch):
    """选题推荐只依赖 DeepSeek：缺 DashScope Key 时也不应报错"""
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "k1")
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", None)
    check_llm_config(require_embedding=False)


def test_check_llm_config_requires_deepseek_even_without_embedding(monkeypatch):
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", None)
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "k2")
    with pytest.raises(HTTPException) as ei:
        check_llm_config(require_embedding=False)
    assert "DEEPSEEK_API_KEY" in ei.value.detail
    assert "DASHSCOPE_API_KEY" not in ei.value.detail

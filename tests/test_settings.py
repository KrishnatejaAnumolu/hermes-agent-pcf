from __future__ import annotations

import pytest

from hermes_pcf.settings import Settings


def test_settings_uses_cf_port_for_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "61000")
    monkeypatch.setenv("API_SERVER_KEY", "a-strong-api-server-key")
    monkeypatch.setenv("LLM_API_KEY", "corp-secret")

    settings = Settings.from_env()

    assert settings.api_server_port == 61000
    assert settings.hermes_llm_base_url == "http://127.0.0.1:8787/v1"
    settings.validate()


def test_settings_rejects_manifest_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_SERVER_KEY", "<replace-with-strong-api-server-key>")
    monkeypatch.setenv("LLM_API_KEY", "corp-secret")

    settings = Settings.from_env()

    with pytest.raises(ValueError, match="API_SERVER_KEY"):
        settings.validate()


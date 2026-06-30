from __future__ import annotations

import os
from pathlib import Path

from hermes_pcf.settings import Settings
from hermes_pcf.start import app_root, build_hermes_config, prepare_environment


def test_agent_config_forces_tool_follow_through() -> None:
    config = build_hermes_config(_settings())

    agent = config["agent"]
    assert agent["tool_use_enforcement"] is True
    assert agent["intent_ack_continuation"] is True
    assert "do not stop after describing the plan" in agent["coding_instructions"][0]
    assert any("hermes_pcf.bitbucket_pr" in item for item in agent["coding_instructions"])
    assert any("JSON tool directive" in item for item in agent["coding_instructions"])
    assert "PYTHONPATH" in config["terminal"]["env_passthrough"]


def test_prepare_environment_makes_app_package_importable(monkeypatch) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)

    prepare_environment(_settings())

    assert app_root() in os.environ["PYTHONPATH"].split(os.pathsep)


def _settings() -> Settings:
    return Settings(
        app_name="test",
        hermes_home=Path(".hermes"),
        hermes_workdir=Path("workspace"),
        api_server_host="0.0.0.0",
        api_server_port=8642,
        api_server_key="a-strong-api-server-key",
        api_server_model_name="hermes-agent",
        api_server_cors_origins="",
        llm_proxy_host="127.0.0.1",
        llm_proxy_port=8787,
        llm_base_url="https://syf-chat-gpt-service.app.uat.pcf.syfbank.com",
        llm_chat_path="/gpt/chat/completions",
        llm_api_key="corp-secret",
        llm_model="GPT-5.2",
        llm_api_version="GPT-5.2",
        syf_channel_id="dise",
        llm_timeout_seconds=1800.0,
        llm_proxy_upstream_streaming=False,
        llm_proxy_force_model=True,
        llm_proxy_strip_model=False,
    )

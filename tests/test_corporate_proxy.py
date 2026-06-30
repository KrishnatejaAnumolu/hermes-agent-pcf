from __future__ import annotations

import json

from hermes_pcf.corporate_proxy import (
    _build_upstream_headers,
    _build_upstream_payload,
    _completion_to_sse_bytes,
)
from hermes_pcf.settings import Settings


def test_build_upstream_payload_forces_model_and_removes_stream() -> None:
    settings = _settings(llm_proxy_upstream_streaming=False)
    payload = {
        "model": "anything",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    upstream_payload = _build_upstream_payload(settings, payload)

    assert upstream_payload["model"] == "GPT-5.2"
    assert "stream" not in upstream_payload
    assert "stream_options" not in upstream_payload


def test_build_upstream_headers_use_corporate_header_names() -> None:
    settings = _settings()

    headers = _build_upstream_headers(settings)

    assert headers["api-key"] == "corp-secret"
    assert headers["api-version"] == "GPT-5.2"
    assert headers["X-SYF-ChannelId"] == "dise"
    assert "Authorization" not in headers


def test_non_streaming_completion_can_be_wrapped_as_openai_sse() -> None:
    settings = _settings()
    completion = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 123,
        "model": "GPT-5.2",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello back"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }

    sse = _completion_to_sse_bytes(settings, json.dumps(completion).encode()).decode()

    assert "data: [DONE]" in sse
    assert '"content":"hello back"' in sse
    assert '"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}' in sse


def _settings(**overrides: object) -> Settings:
    values = dict(
        app_name="test",
        hermes_home=__import__("pathlib").Path(".hermes"),
        hermes_workdir=__import__("pathlib").Path("workspace"),
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
    values.update(overrides)
    return Settings(**values)


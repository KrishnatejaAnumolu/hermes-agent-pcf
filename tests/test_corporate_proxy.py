from __future__ import annotations

import json

from hermes_pcf.corporate_proxy import (
    _completion_to_chat_bytes,
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


def test_build_upstream_payload_repairs_orphan_tool_messages() -> None:
    settings = _settings()
    payload = {
        "model": "anything",
        "messages": [
            {"role": "user", "content": "clone the repo"},
            {
                "role": "tool",
                "tool_call_id": "call_clone",
                "name": "terminal",
                "content": "cloned repo to /home/vcap/app/workspace/repos/EUI/vista",
            },
        ],
        "tools": [{"type": "function", "function": {"name": "terminal"}}],
    }

    upstream_payload = _build_upstream_payload(settings, payload)

    messages = upstream_payload["messages"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] is None
    assert messages[1]["tool_calls"][0]["id"] == "call_clone"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "terminal"
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "call_clone"


def test_build_upstream_payload_keeps_valid_tool_message_sequence() -> None:
    settings = _settings()
    payload = {
        "model": "anything",
        "messages": [
            {"role": "user", "content": "clone the repo"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_clone",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_clone",
                "name": "terminal",
                "content": "cloned repo",
            },
        ],
        "tools": [{"type": "function", "function": {"name": "terminal"}}],
    }

    upstream_payload = _build_upstream_payload(settings, payload)

    assert upstream_payload["messages"] == payload["messages"]


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


def test_json_tool_directive_content_is_wrapped_as_tool_call_sse() -> None:
    settings = _settings()
    request_payload = {
        "tools": [
            {
                "type": "function",
                "function": {"name": "terminal"},
            }
        ]
    }
    completion = {
        "id": "chatcmpl-tool",
        "created": 123,
        "model": "GPT-5.2",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '\n'.join(
                        [
                            '{"tool":"terminal","args":{"cmd":"python -m hermes_pcf.bitbucket_clone https://bitbucket.glb.syfbank.com/scm/eui/vista.git"}}',
                            '{"tool":"terminal","args":{"cmd":"pwd"}}',
                        ]
                    ),
                },
                "finish_reason": "stop",
            }
        ],
    }

    sse = _completion_to_sse_bytes(settings, json.dumps(completion).encode(), request_payload).decode()

    assert '"finish_reason":"tool_calls"' in sse
    assert '"name":"terminal"' in sse
    assert "hermes_pcf.bitbucket_clone" in sse
    assert "pwd" not in sse


def test_json_tool_directive_content_is_wrapped_as_tool_call_chat_response() -> None:
    settings = _settings()
    request_payload = {
        "tools": [
            {
                "type": "function",
                "function": {"name": "terminal"},
            }
        ]
    }
    completion = {
        "id": "chatcmpl-tool",
        "created": 123,
        "model": "GPT-5.2",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"tool":"terminal","args":{"cmd":"pwd"}}',
                },
                "finish_reason": "stop",
            }
        ],
    }

    converted = json.loads(_completion_to_chat_bytes(settings, json.dumps(completion).encode(), request_payload))

    message = converted["choices"][0]["message"]
    assert converted["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "terminal"


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

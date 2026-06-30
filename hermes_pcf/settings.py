from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


PLACEHOLDER_PREFIXES = ("<replace", "replace-", "changeme", "todo")


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None else value.strip()


def _int_env(name: str, default: int) -> int:
    value = _env(name)
    if not value:
        return default
    return int(value)


def _optional_int_env(*names: str) -> int | None:
    for name in names:
        value = _env(name)
        if value:
            return int(value)
    return None


def _float_env(name: str, default: float) -> float:
    value = _env(name)
    if not value:
        return default
    return float(value)


def _bool_env(name: str, default: bool = False) -> bool:
    value = _env(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _csv_env(name: str, default: str) -> list[str]:
    return [item.strip() for item in _env(name, default).split(",") if item.strip()]


def _json_object_env(name: str) -> dict[str, str]:
    value = _env(name)
    if not value:
        return {}
    parsed: Any = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return {str(key): str(val) for key, val in parsed.items()}


def _looks_like_placeholder(value: str) -> bool:
    lower = value.strip().lower()
    return not lower or any(lower.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)


@dataclass(frozen=True)
class Settings:
    app_name: str
    hermes_home: Path
    hermes_workdir: Path
    api_server_host: str
    api_server_port: int
    api_server_key: str
    api_server_model_name: str
    api_server_cors_origins: str
    llm_proxy_host: str
    llm_proxy_port: int
    llm_base_url: str
    llm_chat_path: str
    llm_api_key: str
    llm_model: str
    llm_api_version: str
    syf_channel_id: str
    llm_timeout_seconds: float
    llm_proxy_upstream_streaming: bool
    llm_proxy_force_model: bool
    llm_proxy_strip_model: bool
    llm_extra_headers: dict[str, str] = field(default_factory=dict)
    hermes_context_length: int = 128000
    hermes_model_max_tokens: int | None = 8192
    hermes_max_turns: int = 60
    hermes_reasoning_effort: str = "medium"
    hermes_memory_enabled: bool = False
    hermes_user_profile_enabled: bool = False
    hermes_api_server_toolsets: list[str] = field(default_factory=lambda: ["web", "file", "skills", "todo"])
    terminal_timeout: int = 180
    tirith_enabled: bool = False
    tirith_fail_open: bool = True
    overwrite_config: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        llm_model = _env("LLM_MODEL", "GPT-5.2")
        return cls(
            app_name=_env("APP_NAME", "hermes-agent-pcf"),
            hermes_home=Path(_env("HERMES_HOME", ".hermes")),
            hermes_workdir=Path(_env("HERMES_WORKDIR", "workspace")),
            api_server_host=_env("API_SERVER_HOST", "0.0.0.0"),
            api_server_port=int(_env("PORT", _env("API_SERVER_PORT", "8642"))),
            api_server_key=_env("API_SERVER_KEY"),
            api_server_model_name=_env("API_SERVER_MODEL_NAME", "hermes-agent"),
            api_server_cors_origins=_env("API_SERVER_CORS_ORIGINS"),
            llm_proxy_host=_env("LLM_PROXY_HOST", "127.0.0.1"),
            llm_proxy_port=_int_env("LLM_PROXY_PORT", 8787),
            llm_base_url=_env("LLM_BASE_URL", "https://syf-chat-gpt-service.app.uat.pcf.syfbank.com"),
            llm_chat_path=_env("LLM_CHAT_PATH", "/gpt/chat/completions"),
            llm_api_key=_env("LLM_API_KEY"),
            llm_model=llm_model,
            llm_api_version=_env("LLM_API_VERSION", llm_model),
            syf_channel_id=_env("SYF_CHANNEL_ID", "dise"),
            llm_timeout_seconds=_float_env("LLM_TIMEOUT_SECONDS", 1800.0),
            llm_proxy_upstream_streaming=_bool_env("LLM_PROXY_UPSTREAM_STREAMING", False),
            llm_proxy_force_model=_bool_env("LLM_PROXY_FORCE_MODEL", True),
            llm_proxy_strip_model=_bool_env("LLM_PROXY_STRIP_MODEL", False),
            llm_extra_headers=_json_object_env("LLM_EXTRA_HEADERS"),
            hermes_context_length=_int_env("HERMES_MODEL_CONTEXT_LENGTH", 128000),
            hermes_model_max_tokens=_optional_int_env("HERMES_MODEL_MAX_TOKENS", "HERMES_MAX_TOKENS"),
            hermes_max_turns=_int_env("HERMES_MAX_TURNS", 60),
            hermes_reasoning_effort=_env("HERMES_REASONING_EFFORT", "medium"),
            hermes_memory_enabled=_bool_env("HERMES_MEMORY_ENABLED", False),
            hermes_user_profile_enabled=_bool_env("HERMES_USER_PROFILE_ENABLED", False),
            hermes_api_server_toolsets=_csv_env("HERMES_API_SERVER_TOOLSETS", "web,file,skills,todo"),
            terminal_timeout=_int_env("TERMINAL_TIMEOUT", 180),
            tirith_enabled=_bool_env("TIRITH_ENABLED", False),
            tirith_fail_open=_bool_env("TIRITH_FAIL_OPEN", True),
            overwrite_config=_bool_env("HERMES_OVERWRITE_CONFIG", True),
        )

    @property
    def llm_chat_url(self) -> str:
        return urljoin(self.llm_base_url.rstrip("/") + "/", self.llm_chat_path.lstrip("/"))

    @property
    def hermes_llm_base_url(self) -> str:
        return f"http://{self.llm_proxy_host}:{self.llm_proxy_port}/v1"

    def validate(self) -> None:
        if _looks_like_placeholder(self.api_server_key) or len(self.api_server_key) < 16:
            raise ValueError("API_SERVER_KEY must be set to a strong value at least 16 characters long")
        if _looks_like_placeholder(self.llm_api_key):
            raise ValueError("LLM_API_KEY must be set in the manifest before deployment")
        if not self.llm_base_url.startswith(("http://", "https://")):
            raise ValueError("LLM_BASE_URL must start with http:// or https://")
        if not self.llm_chat_path.startswith("/"):
            raise ValueError("LLM_CHAT_PATH must start with /")
        if not self.llm_model:
            raise ValueError("LLM_MODEL must be set")

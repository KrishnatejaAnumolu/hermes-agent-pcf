from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import yaml
from aiohttp import web

from .corporate_proxy import create_app
from .settings import Settings

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_hermes_config(settings: Settings) -> dict[str, Any]:
    model: dict[str, Any] = {
        "default": settings.llm_model,
        "provider": "custom",
        "base_url": settings.hermes_llm_base_url,
        "api_key": "local-proxy-key",
        "api_mode": "chat",
        "context_length": settings.hermes_context_length,
        "default_headers": {
            "X-Hermes-PCF-Proxy": "true",
        },
    }
    if settings.hermes_model_max_tokens is not None:
        model["max_tokens"] = settings.hermes_model_max_tokens

    return {
        "model": model,
        "platform_toolsets": {
            "api_server": settings.hermes_api_server_toolsets,
        },
        "max_turns": settings.hermes_max_turns,
        "reasoning": {"effort": settings.hermes_reasoning_effort},
        "memory": {"enabled": settings.hermes_memory_enabled},
        "user_profile": {"enabled": settings.hermes_user_profile_enabled},
        "security": {
            "tirith_enabled": settings.tirith_enabled,
            "tirith_fail_open": settings.tirith_fail_open,
        },
        "terminal": {
            "default_workdir": str(settings.hermes_workdir),
            "timeout": settings.terminal_timeout,
        },
        "gateway": {
            "platforms": {
                "api_server": {
                    "enabled": True,
                    "extra": {
                        "host": settings.api_server_host,
                        "port": settings.api_server_port,
                        "key": settings.api_server_key,
                        "model_name": settings.api_server_model_name,
                        "cors_origins": _cors_origins(settings),
                    },
                },
            }
        },
    }


def _cors_origins(settings: Settings) -> list[str]:
    return [origin.strip() for origin in settings.api_server_cors_origins.split(",") if origin.strip()]


def write_hermes_config(settings: Settings) -> None:
    settings.hermes_home.mkdir(parents=True, exist_ok=True)
    settings.hermes_workdir.mkdir(parents=True, exist_ok=True)
    config_path = settings.hermes_home / "config.yaml"

    if config_path.exists() and not settings.overwrite_config:
        LOGGER.info("Using existing Hermes config at %s", config_path)
        return

    config = build_hermes_config(settings)
    tmp_path = config_path.with_suffix(".yaml.tmp")
    tmp_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    tmp_path.replace(config_path)
    LOGGER.info("Wrote Hermes config to %s", config_path)


def prepare_environment(settings: Settings) -> None:
    os.environ["HERMES_HOME"] = str(settings.hermes_home)
    os.environ["API_SERVER_ENABLED"] = "true"
    os.environ["API_SERVER_HOST"] = settings.api_server_host
    os.environ["API_SERVER_PORT"] = str(settings.api_server_port)
    os.environ["API_SERVER_KEY"] = settings.api_server_key
    os.environ["API_SERVER_MODEL_NAME"] = settings.api_server_model_name
    os.environ["TIRITH_ENABLED"] = str(settings.tirith_enabled).lower()
    os.environ["TIRITH_FAIL_OPEN"] = str(settings.tirith_fail_open).lower()
    os.environ.setdefault("HERMES_API_TIMEOUT", str(int(settings.llm_timeout_seconds)))
    os.environ.setdefault("HERMES_STREAM_READ_TIMEOUT", str(max(300, int(settings.llm_timeout_seconds))))


async def start_proxy(settings: Settings) -> web.AppRunner:
    app = create_app(settings)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, settings.llm_proxy_host, settings.llm_proxy_port)
    await site.start()
    LOGGER.info("Started local LLM proxy at %s", settings.hermes_llm_base_url)
    return runner


async def main_async() -> None:
    settings = Settings.from_env()
    settings.validate()
    prepare_environment(settings)
    write_hermes_config(settings)
    proxy_runner = await start_proxy(settings)

    try:
        from gateway.run import start_gateway

        ok = await start_gateway(replace=True, verbosity=0)
        if not ok:
            raise RuntimeError("Hermes gateway exited unsuccessfully")
    finally:
        await proxy_runner.cleanup()


def main() -> None:
    configure_logging()
    try:
        asyncio.run(main_async())
    except Exception:
        LOGGER.exception("Hermes PCF wrapper failed")
        raise


if __name__ == "__main__":
    main()

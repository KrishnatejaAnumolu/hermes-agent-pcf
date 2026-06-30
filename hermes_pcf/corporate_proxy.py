from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

from .settings import Settings

LOGGER = logging.getLogger(__name__)


def create_app(settings: Settings) -> web.Application:
    app = web.Application(client_max_size=20 * 1024 * 1024)
    app["settings"] = settings
    app.cleanup_ctx.append(_client_session_context)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/chat/completions", chat_completions)
    return app


async def _client_session_context(app: web.Application) -> AsyncIterator[None]:
    settings: Settings = app["settings"]
    timeout = ClientTimeout(
        total=settings.llm_timeout_seconds,
        connect=30,
        sock_connect=30,
        sock_read=settings.llm_timeout_seconds,
    )
    app["client"] = ClientSession(timeout=timeout)
    try:
        yield
    finally:
        await app["client"].close()


async def health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "hermes-corporate-llm-proxy"})


async def models(request: web.Request) -> web.Response:
    settings: Settings = request.app["settings"]
    return web.json_response(
        {
            "object": "list",
            "data": [
                {
                    "id": settings.llm_model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "corporate-proxy",
                }
            ],
        }
    )


async def chat_completions(request: web.Request) -> web.StreamResponse:
    settings: Settings = request.app["settings"]
    client: ClientSession = request.app["client"]
    payload = await request.json()
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="JSON body must be an object")

    requested_stream = _as_bool(payload.get("stream"))
    upstream_payload = _build_upstream_payload(settings, payload)
    headers = _build_upstream_headers(settings, request)
    upstream_streaming = requested_stream and settings.llm_proxy_upstream_streaming

    LOGGER.info(
        "Forwarding chat completion to corporate proxy stream=%s upstream_stream=%s tracking_id=%s",
        requested_stream,
        upstream_streaming,
        headers["X-SYF-Request-TrackingId"],
    )

    async with client.post(
        settings.llm_chat_url,
        headers=headers,
        json=upstream_payload,
    ) as upstream_response:
        if upstream_response.status >= 400:
            body = await upstream_response.read()
            return web.Response(
                status=upstream_response.status,
                body=body,
                headers=_safe_response_headers(upstream_response.headers),
            )

        if upstream_streaming:
            return await _stream_upstream_sse(request, upstream_response)

        body = await upstream_response.read()
        if requested_stream:
            return web.Response(
                body=_completion_to_sse_bytes(settings, body),
                content_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        return web.Response(
            body=body,
            headers=_safe_response_headers(upstream_response.headers),
        )


def _build_upstream_payload(settings: Settings, payload: dict[str, Any]) -> dict[str, Any]:
    upstream_payload = dict(payload)

    if settings.llm_proxy_strip_model:
        upstream_payload.pop("model", None)
    elif settings.llm_proxy_force_model:
        upstream_payload["model"] = settings.llm_model

    if not settings.llm_proxy_upstream_streaming:
        upstream_payload.pop("stream", None)
        upstream_payload.pop("stream_options", None)

    return upstream_payload


def _build_upstream_headers(settings: Settings, request: web.Request | None = None) -> dict[str, str]:
    tracking_id = str(uuid.uuid4())
    if request is not None:
        tracking_id = (
            request.headers.get("X-SYF-Request-TrackingId")
            or request.headers.get("X-Request-Id")
            or request.headers.get("X-Correlation-Id")
            or tracking_id
        )

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if settings.llm_proxy_upstream_streaming else "application/json",
        "api-key": settings.llm_api_key,
        "api-version": settings.llm_api_version,
        "X-SYF-ChannelId": settings.syf_channel_id,
        "X-SYF-Request-TrackingId": tracking_id,
    }
    headers.update(settings.llm_extra_headers)
    return headers


def _safe_response_headers(headers: Any) -> dict[str, str]:
    safe = {}
    for key in ("Content-Type", "Cache-Control", "X-Request-Id", "X-Correlation-Id"):
        value = headers.get(key)
        if value:
            safe[key] = value
    return safe


async def _stream_upstream_sse(request: web.Request, upstream_response: Any) -> web.StreamResponse:
    response = web.StreamResponse(
        status=upstream_response.status,
        headers={
            "Content-Type": upstream_response.headers.get("Content-Type", "text/event-stream"),
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)
    async for chunk in upstream_response.content.iter_chunked(8192):
        await response.write(chunk)
    await response.write_eof()
    return response


def _completion_to_sse_bytes(settings: Settings, body: bytes) -> bytes:
    if body.lstrip().startswith(b"data:"):
        return body

    completion = json.loads(body.decode("utf-8"))
    events = list(_completion_to_sse_events(settings, completion))
    events.append("data: [DONE]\n\n")
    return "".join(events).encode("utf-8")


def _completion_to_sse_events(settings: Settings, completion: dict[str, Any]) -> AsyncIterator[str] | list[str]:
    completion_id = str(completion.get("id") or f"chatcmpl-{uuid.uuid4().hex}")
    model = str(completion.get("model") or settings.llm_model)
    created = int(completion.get("created") or time.time())
    choices = completion.get("choices") or []

    events: list[str] = []
    for index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or choice.get("delta") or {}
        if not isinstance(message, dict):
            message = {"content": str(message)}

        role = message.get("role") or "assistant"
        events.append(_sse_event(_chunk(completion_id, model, created, index, {"role": role}, None)))

        content = message.get("content")
        if content:
            events.append(_sse_event(_chunk(completion_id, model, created, index, {"content": content}, None)))

        for tool_index, tool_call in enumerate(message.get("tool_calls") or []):
            events.append(
                _sse_event(
                    _chunk(
                        completion_id,
                        model,
                        created,
                        index,
                        {"tool_calls": [dict(tool_call, index=tool_call.get("index", tool_index))]},
                        None,
                    )
                )
            )

        finish_reason = choice.get("finish_reason") or "stop"
        events.append(_sse_event(_chunk(completion_id, model, created, index, {}, finish_reason)))

    usage = completion.get("usage")
    if usage:
        events.append(
            _sse_event(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": usage,
                }
            )
        )

    return events


def _chunk(
    completion_id: str,
    model: str,
    created: int,
    index: int,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}],
    }


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


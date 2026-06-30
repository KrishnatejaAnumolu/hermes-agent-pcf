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
                body=_completion_to_sse_bytes(settings, body, payload),
                content_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        return web.Response(
            body=_completion_to_chat_bytes(settings, body, payload),
            headers=_safe_response_headers(upstream_response.headers),
        )


def _build_upstream_payload(settings: Settings, payload: dict[str, Any]) -> dict[str, Any]:
    upstream_payload = dict(payload)
    upstream_payload["messages"] = _repair_orphan_tool_messages(
        upstream_payload.get("messages"),
        _request_tool_names(upstream_payload),
    )

    if settings.llm_proxy_strip_model:
        upstream_payload.pop("model", None)
    elif settings.llm_proxy_force_model:
        upstream_payload["model"] = settings.llm_model

    if not settings.llm_proxy_upstream_streaming:
        upstream_payload.pop("stream", None)
        upstream_payload.pop("stream_options", None)

    return upstream_payload


def _repair_orphan_tool_messages(messages: Any, allowed_tool_names: set[str]) -> Any:
    if not isinstance(messages, list):
        return messages

    repaired_count = 0
    normalized: list[Any] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not _is_tool_message(message):
            normalized.append(message)
            index += 1
            continue

        group: list[dict[str, Any]] = []
        while index < len(messages) and _is_tool_message(messages[index]):
            group.append(_normalized_tool_message(messages[index], len(normalized) + len(group)))
            index += 1

        previous = normalized[-1] if normalized else None
        if not _assistant_covers_tool_group(previous, group):
            normalized.append(_synthetic_assistant_for_tool_group(group, allowed_tool_names))
            repaired_count += len(group)

        normalized.extend(group)

    if repaired_count:
        LOGGER.warning(
            "Inserted synthetic assistant tool_calls before %s orphan tool message(s)",
            repaired_count,
        )

    return normalized


def _is_tool_message(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") == "tool"


def _normalized_tool_message(message: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    normalized = dict(message)
    tool_call_id = normalized.get("tool_call_id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        normalized["tool_call_id"] = f"call_repaired_{fallback_index}_{uuid.uuid4().hex[:12]}"
    return normalized


def _assistant_covers_tool_group(previous: Any, group: list[dict[str, Any]]) -> bool:
    if not isinstance(previous, dict) or previous.get("role") != "assistant":
        return False

    tool_calls = previous.get("tool_calls")
    if not isinstance(tool_calls, list):
        return False

    assistant_ids = {call.get("id") for call in tool_calls if isinstance(call, dict)}
    return all(message.get("tool_call_id") in assistant_ids for message in group)


def _synthetic_assistant_for_tool_group(
    group: list[dict[str, Any]],
    allowed_tool_names: set[str],
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": str(message["tool_call_id"]),
                "type": "function",
                "function": {
                    "name": _tool_name_for_repair(message, allowed_tool_names),
                    "arguments": "{}",
                },
            }
            for message in group
        ],
    }


def _tool_name_for_repair(message: dict[str, Any], allowed_tool_names: set[str]) -> str:
    name = message.get("name")
    if isinstance(name, str) and (not allowed_tool_names or name in allowed_tool_names):
        return name
    if "terminal" in allowed_tool_names:
        return "terminal"
    if allowed_tool_names:
        return sorted(allowed_tool_names)[0]
    return "terminal"


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


def _completion_to_chat_bytes(
    settings: Settings,
    body: bytes,
    request_payload: dict[str, Any] | None = None,
) -> bytes:
    if body.lstrip().startswith(b"data:"):
        return body

    completion = json.loads(body.decode("utf-8"))
    completion = _completion_with_json_tool_calls(settings, completion, request_payload)
    return json.dumps(completion, separators=(",", ":")).encode("utf-8")


def _completion_to_sse_bytes(
    settings: Settings,
    body: bytes,
    request_payload: dict[str, Any] | None = None,
) -> bytes:
    if body.lstrip().startswith(b"data:"):
        return body

    completion = json.loads(body.decode("utf-8"))
    completion = _completion_with_json_tool_calls(settings, completion, request_payload)
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


def _completion_with_json_tool_calls(
    settings: Settings,
    completion: dict[str, Any],
    request_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if not settings.llm_proxy_json_tool_calls or request_payload is None:
        return completion

    allowed_tool_names = _request_tool_names(request_payload)
    if not allowed_tool_names:
        return completion

    choices = completion.get("choices")
    if not isinstance(choices, list):
        return completion

    changed = False
    converted_count = 0
    converted_choices: list[Any] = []
    for choice in choices:
        if not isinstance(choice, dict):
            converted_choices.append(choice)
            continue

        message_key = "message" if isinstance(choice.get("message"), dict) else "delta"
        message = choice.get(message_key)
        if not isinstance(message, dict) or message.get("tool_calls"):
            converted_choices.append(choice)
            continue

        tool_calls = _json_tool_calls_from_content(
            message.get("content"),
            allowed_tool_names,
            max(0, settings.llm_proxy_json_tool_call_max),
        )
        if not tool_calls:
            converted_choices.append(choice)
            continue

        converted_message = dict(message)
        converted_message["role"] = converted_message.get("role") or "assistant"
        converted_message["content"] = None
        converted_message["tool_calls"] = tool_calls

        converted_choice = dict(choice)
        converted_choice[message_key] = converted_message
        converted_choice["finish_reason"] = "tool_calls"
        converted_choices.append(converted_choice)
        changed = True
        converted_count += len(tool_calls)

    if not changed:
        return completion

    LOGGER.warning(
        "Converted %s JSON tool directive(s) from model text into structured tool_calls",
        converted_count,
    )
    converted_completion = dict(completion)
    converted_completion["choices"] = converted_choices
    return converted_completion


def _request_tool_names(request_payload: dict[str, Any]) -> set[str]:
    tool_names: set[str] = set()
    tools = request_payload.get("tools")
    if not isinstance(tools, list):
        return tool_names

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            tool_names.add(function["name"])
            continue
        if isinstance(tool.get("name"), str):
            tool_names.add(tool["name"])

    return tool_names


def _json_tool_calls_from_content(content: Any, allowed_tool_names: set[str], max_calls: int) -> list[dict[str, Any]]:
    if max_calls <= 0 or not isinstance(content, str):
        return []

    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    index = 0
    while index < len(content):
        index = _skip_tool_directive_separators(content, index)
        if index >= len(content):
            return calls
        if content[index] not in "{[":
            return calls

        try:
            parsed, index = decoder.raw_decode(content, index)
        except json.JSONDecodeError:
            return calls

        directive = _parse_tool_directive_value(parsed, allowed_tool_names)
        if directive is None:
            return calls

        calls.append(_openai_tool_call(directive[0], directive[1]))
        if len(calls) >= max_calls:
            return calls

    return calls


def _skip_tool_directive_separators(content: str, index: int) -> int:
    while index < len(content) and content[index].isspace():
        index += 1

    if content.startswith("```", index):
        index += 3
        while index < len(content) and content[index] not in "\r\n":
            index += 1
        while index < len(content) and content[index] in "\r\n":
            index += 1

    return index


def _parse_tool_directive(line: str, allowed_tool_names: set[str]) -> tuple[str, dict[str, Any]] | None:
    if not line.startswith(("{", "[")):
        return None

    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None

    return _parse_tool_directive_value(parsed, allowed_tool_names)


def _parse_tool_directive_value(parsed: Any, allowed_tool_names: set[str]) -> tuple[str, dict[str, Any]] | None:
    if isinstance(parsed, list):
        if not parsed or not isinstance(parsed[0], dict):
            return None
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return None

    name = parsed.get("tool") or parsed.get("name")
    if not isinstance(name, str) or name not in allowed_tool_names:
        return None

    args = parsed.get("args", parsed.get("arguments", {}))
    if isinstance(args, str):
        try:
            parsed_args = json.loads(args)
        except json.JSONDecodeError:
            parsed_args = {"input": args}
        args = parsed_args
    if not isinstance(args, dict):
        args = {"input": args}

    args = _normalize_tool_arguments(name, args)
    return name, args


def _normalize_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name != "terminal":
        return arguments

    normalized = dict(arguments)
    if "command" not in normalized and "cmd" in normalized:
        normalized["command"] = normalized.pop("cmd")
    if "workdir" not in normalized:
        for alias in ("cwd", "working_directory"):
            if alias in normalized:
                normalized["workdir"] = normalized.pop(alias)
                break
    if "timeout" not in normalized and "timeout_ms" in normalized:
        timeout_ms = normalized.pop("timeout_ms")
        try:
            normalized["timeout"] = max(1, int(timeout_ms) // 1000)
        except (TypeError, ValueError):
            pass
    return normalized


def _openai_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, separators=(",", ":")),
        },
    }


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

# Hermes Agent on PCF

This project deploys [Nous Research Hermes Agent](https://github.com/nousresearch/hermes-agent) to Pivotal Cloud Foundry with the standard Python buildpack.

Hermes talks to a local OpenAI-compatible proxy at `http://127.0.0.1:8787/v1`. That proxy forwards chat-completion requests to the corporate GPT 5.2 endpoint with the required `api-key`, `api-version`, `X-SYF-ChannelId`, and `X-SYF-Request-TrackingId` headers.

## Files

- `manifest.yml` deploys a routed Hermes API server.
- `manifest-worker.yml` deploys the same process without a route for future worker or messaging integrations.
- `Procfile` starts `python -m hermes_pcf.start`.
- `hermes_pcf/start.py` writes the runtime Hermes config and launches Hermes.
- `hermes_pcf/corporate_proxy.py` translates OpenAI-style requests into the corporate proxy format.
- `vendor/` contains the pinned Hermes wheel and Linux Python 3.11 dependency wheels so PCF staging does not need GitHub or PyPI egress.

The `requirements.txt` file intentionally uses `--no-index` and `--find-links vendor`. This avoids staging failures in PCF foundations where outbound GitHub access is blocked or unstable.

## Configure

Edit `manifest.yml` before pushing:

```yaml
API_SERVER_KEY: "<replace-with-strong-api-server-key>"
LLM_API_KEY: "<replace-with-corporate-llm-api-key>"
LLM_BASE_URL: https://syf-chat-gpt-service.app.uat.pcf.syfbank.com
LLM_CHAT_PATH: /gpt/chat/completions
LLM_MODEL: GPT-5.2
LLM_API_VERSION: GPT-5.2
SYF_CHANNEL_ID: dise
```

`API_SERVER_KEY` protects the public Hermes API. `LLM_API_KEY` is only used server-side when the local proxy calls the corporate GPT service.

By default `LLM_PROXY_UPSTREAM_STREAMING` is `false`. Hermes can still ask for streaming; this proxy sends a normal corporate request and converts the JSON response into OpenAI-compatible server-sent events for Hermes. Set it to `true` only if the corporate endpoint supports OpenAI streaming responses.

Some corporate GPT proxies return tool-call directives as assistant text instead of OpenAI `tool_calls`. This project enables `LLM_PROXY_JSON_TOOL_CALLS` by default so a leading line like `{"tool":"terminal","args":{"cmd":"..."}}` is converted back into a structured tool call for Hermes. `LLM_PROXY_JSON_TOOL_CALL_MAX` defaults to `1` so multi-step terminal work happens one completed command at a time.

## Deploy

```bash
cd hermes-agent-pcf
cf login -a https://api.<your-pcf-foundation>
cf target -o <org> -s <space>
cf push -f manifest.yml
```

Check health:

```bash
cf app hermes-agent-pcf
curl https://<route>/health
```

Call the Hermes OpenAI-compatible endpoint:

```bash
curl https://<route>/v1/chat/completions \
  -H "Authorization: Bearer <API_SERVER_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hermes-agent",
    "messages": [
      {"role": "user", "content": "Say hello from Hermes on PCF"}
    ]
  }'
```

List the exposed model:

```bash
curl https://<route>/v1/models \
  -H "Authorization: Bearer <API_SERVER_KEY>"
```

## Environment Reference

| Variable | Purpose |
| --- | --- |
| `API_SERVER_KEY` | Bearer token required by the public Hermes API. Must be at least 16 characters. |
| `API_SERVER_MODEL_NAME` | Model name exposed by Hermes, default `hermes-agent`. |
| `PYTHONPATH` | Includes `/home/vcap/app` so terminal tools can run packaged helpers from the workspace directory. |
| `LLM_BASE_URL` | Corporate proxy origin, without the chat path. |
| `LLM_CHAT_PATH` | Corporate chat-completions path, default `/gpt/chat/completions`. |
| `LLM_API_KEY` | Corporate LLM API key, sent as `api-key`. |
| `LLM_MODEL` | Corporate model identifier, default `GPT-5.2`. |
| `LLM_API_VERSION` | Header value sent as `api-version`; defaults to `LLM_MODEL`. |
| `SYF_CHANNEL_ID` | Header value sent as `X-SYF-ChannelId`. |
| `LLM_PROXY_UPSTREAM_STREAMING` | Whether the corporate endpoint supports streaming SSE directly. |
| `LLM_PROXY_FORCE_MODEL` | Replaces inbound `model` with `LLM_MODEL` before forwarding. |
| `LLM_PROXY_STRIP_MODEL` | Removes `model` from the forwarded payload if the corporate endpoint rejects it. |
| `LLM_PROXY_JSON_TOOL_CALLS` | Converts leading JSON tool directives returned as text into OpenAI `tool_calls`. |
| `LLM_PROXY_JSON_TOOL_CALL_MAX` | Maximum converted tool directives per model response, default `1`. |
| `LLM_EXTRA_HEADERS` | Optional JSON object of extra headers to add to corporate requests. |
| `HERMES_API_SERVER_TOOLSETS` | Comma-separated Hermes API toolsets exposed in API-server mode. |
| `TIRITH_ENABLED` | Defaults to `false` to avoid runtime GitHub downloads in locked-down PCF spaces. Set to `true` if the helper is available. |
| `BITBUCKET_SERVER_URL` | Bitbucket Server origin, default `https://bitbucket.glb.syfbank.com`. |
| `BITBUCKET_SERVER_BEARER_TOKEN` | Bearer token used by the Bitbucket clone helper. |
| `BITBUCKET_ALLOWED_PROJECTS` | Comma-separated project allowlist for clone helper, default `EUI`. |
| `BITBUCKET_WORKDIR` | Directory where Bitbucket repos are cloned. |

## Bitbucket Repos

The API server enables Hermes terminal tools and includes a helper for Bitbucket Server URLs:

```bash
python -m hermes_pcf.bitbucket_clone https://bitbucket.glb.syfbank.com/projects/EUI/repos/vista/
python -m hermes_pcf.bitbucket_clone https://bitbucket.glb.syfbank.com/projects/EUI/repos/vista/ --branch develop
python -m hermes_pcf.bitbucket_pr https://bitbucket.glb.syfbank.com/projects/EUI/repos/vista/pull-requests/2331/overview
```

The helper converts Bitbucket Server web URLs to `/scm/<project>/<repo>.git`, injects the bearer token into the Git subprocess without writing it to global Git config, and clones or updates under `BITBUCKET_WORKDIR`.
The PR helper calls Bitbucket Server REST API with the same bearer token and prints JSON containing the PR title, description, source/target refs, author, and reviewers.

Example prompt:

```text
Clone https://bitbucket.glb.syfbank.com/projects/EUI/repos/vista/ and explain how login routing works.
```

## Local Smoke Test

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

To run locally, export real keys first:

```bash
export API_SERVER_KEY="local-api-server-key-with-length"
export LLM_API_KEY="<corporate-key>"
python -m hermes_pcf.start
```

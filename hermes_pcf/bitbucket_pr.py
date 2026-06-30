from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .bitbucket_clone import SAFE_KEY_RE


@dataclass(frozen=True)
class PullRequestRef:
    project: str
    repo: str
    pull_request_id: int
    api_url: str


def parse_pull_request_url(pr_url: str, server_url: str, allowed_projects: list[str]) -> PullRequestRef:
    parsed_input = urlparse(pr_url.rstrip("/"))
    parsed_server = urlparse(server_url.rstrip("/"))
    if parsed_input.scheme not in {"http", "https"} or not parsed_input.netloc:
        raise ValueError("Pull request URL must be an absolute http(s) URL")
    if parsed_input.netloc.lower() != parsed_server.netloc.lower():
        raise ValueError(f"Pull request host must be {parsed_server.netloc}")

    parts = [part for part in parsed_input.path.split("/") if part]
    project = repo = pr_id = ""

    if len(parts) >= 6 and parts[0].lower() == "projects" and parts[2].lower() == "repos":
        project = parts[1]
        repo = parts[3]
        if parts[4].lower() == "pull-requests":
            pr_id = parts[5]
    elif (
        len(parts) >= 9
        and parts[0].lower() == "rest"
        and parts[1].lower() == "api"
        and parts[3].lower() == "projects"
        and parts[5].lower() == "repos"
        and parts[7].lower() == "pull-requests"
    ):
        project = parts[4]
        repo = parts[6]
        pr_id = parts[8]

    if not project or not repo or not pr_id:
        raise ValueError("Expected a Bitbucket Server PR URL like /projects/KEY/repos/repo/pull-requests/123")

    project = project.upper()
    repo = repo.strip()
    _validate_key("project", project)
    _validate_key("repo", repo)
    try:
        pull_request_id = int(pr_id)
    except ValueError as exc:
        raise ValueError(f"Invalid pull request id: {pr_id!r}") from exc
    if pull_request_id <= 0:
        raise ValueError(f"Invalid pull request id: {pr_id!r}")

    allowed = {item.upper() for item in allowed_projects if item}
    if allowed and project not in allowed:
        raise ValueError(f"Project {project!r} is not allowed. Allowed projects: {', '.join(sorted(allowed))}")

    api_url = f"{server_url.rstrip('/')}/rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{pull_request_id}"
    return PullRequestRef(project=project, repo=repo, pull_request_id=pull_request_id, api_url=api_url)


def fetch_pull_request(ref: PullRequestRef, token: str, timeout: int = 60) -> dict[str, Any]:
    if not token or token.startswith("<replace"):
        raise ValueError("BITBUCKET_SERVER_BEARER_TOKEN is not configured")

    request = Request(
        ref.api_url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "hermes-agent-pcf",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Bitbucket API returned HTTP {exc.code}: {body[:2000]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Bitbucket API request failed: {exc.reason}") from exc

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("Bitbucket API returned a non-object JSON response")
    return parsed


def summarize_pull_request(ref: PullRequestRef, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "project": ref.project,
        "repo": ref.repo,
        "pull_request_id": ref.pull_request_id,
        "api_url": ref.api_url,
        "title": payload.get("title") or "",
        "description": payload.get("description") or "",
        "state": payload.get("state") or "",
        "open": payload.get("open"),
        "closed": payload.get("closed"),
        "from_ref": _ref_summary(payload.get("fromRef")),
        "to_ref": _ref_summary(payload.get("toRef")),
        "author": _participant_summary(payload.get("author")),
        "reviewers": [_participant_summary(item) for item in payload.get("reviewers") or []],
        "links": payload.get("links") or {},
    }


def _ref_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "id": value.get("id") or "",
        "display_id": value.get("displayId") or "",
        "latest_commit": value.get("latestCommit") or "",
        "repository": _repo_summary(value.get("repository")),
    }


def _repo_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    project = value.get("project") if isinstance(value.get("project"), dict) else {}
    return {
        "slug": value.get("slug") or "",
        "name": value.get("name") or "",
        "project_key": project.get("key") or "",
    }


def _participant_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    user = value.get("user") if isinstance(value.get("user"), dict) else {}
    return {
        "name": user.get("name") or "",
        "display_name": user.get("displayName") or "",
        "email_address": user.get("emailAddress") or "",
        "role": value.get("role") or "",
        "approved": value.get("approved"),
        "status": value.get("status") or "",
    }


def _validate_key(label: str, value: str) -> None:
    if not value or not SAFE_KEY_RE.match(value):
        raise ValueError(f"Invalid {label}: {value!r}")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Bitbucket Server pull request metadata for Hermes.")
    parser.add_argument("pull_request_url", help="Bitbucket Server pull request URL")
    parser.add_argument("--raw", action="store_true", help="Print the raw Bitbucket API JSON response")
    args = parser.parse_args(argv)

    server_url = os.environ.get("BITBUCKET_SERVER_URL", "https://bitbucket.glb.syfbank.com")
    token = os.environ.get("BITBUCKET_SERVER_BEARER_TOKEN", "")
    allowed_projects = _csv(os.environ.get("BITBUCKET_ALLOWED_PROJECTS", "EUI"))

    try:
        ref = parse_pull_request_url(args.pull_request_url, server_url, allowed_projects)
        payload = fetch_pull_request(ref, token)
        result = payload if args.raw else summarize_pull_request(ref, payload)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class RepoRef:
    project: str
    repo: str
    clone_url: str


def parse_repo_url(repo_url: str, server_url: str, allowed_projects: list[str]) -> RepoRef:
    parsed_input = urlparse(repo_url.rstrip("/"))
    parsed_server = urlparse(server_url.rstrip("/"))
    if parsed_input.scheme not in {"http", "https"} or not parsed_input.netloc:
        raise ValueError("Repository URL must be an absolute http(s) URL")
    if parsed_input.netloc.lower() != parsed_server.netloc.lower():
        raise ValueError(f"Repository host must be {parsed_server.netloc}")

    parts = [part for part in parsed_input.path.split("/") if part]
    project = repo = ""

    if len(parts) >= 4 and parts[0].lower() == "projects" and parts[2].lower() == "repos":
        project = parts[1]
        repo = parts[3]
    elif len(parts) >= 3 and parts[0].lower() == "scm":
        project = parts[1]
        repo = parts[2][:-4] if parts[2].endswith(".git") else parts[2]
    else:
        raise ValueError("Expected a Bitbucket Server URL like /projects/KEY/repos/repo or /scm/key/repo.git")

    project = project.upper()
    repo = repo.strip()
    _validate_key("project", project)
    _validate_key("repo", repo)

    allowed = {item.upper() for item in allowed_projects if item}
    if allowed and project.upper() not in allowed:
        raise ValueError(f"Project {project!r} is not allowed. Allowed projects: {', '.join(sorted(allowed))}")

    base = server_url.rstrip("/")
    clone_url = f"{base}/scm/{project.lower()}/{repo}.git"
    return RepoRef(project=project, repo=repo, clone_url=clone_url)


def clone_or_update(repo_ref: RepoRef, workdir: Path, token: str, branch: str = "") -> dict[str, str]:
    if not token or token.startswith("<replace"):
        raise ValueError("BITBUCKET_SERVER_BEARER_TOKEN is not configured")

    repo_dir = workdir / repo_ref.project / repo_ref.repo
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    if (repo_dir / ".git").exists():
        _git(["remote", "set-url", "origin", repo_ref.clone_url], repo_dir, repo_ref.clone_url, token)
        _git(["fetch", "--all", "--prune"], repo_dir, repo_ref.clone_url, token)
        action = "updated"
    else:
        _git(["clone", repo_ref.clone_url, str(repo_dir)], None, repo_ref.clone_url, token)
        action = "cloned"

    if branch:
        _validate_branch(branch)
        checkout = _git(["checkout", branch], repo_dir, repo_ref.clone_url, token, check=False)
        if checkout.returncode != 0:
            _git(["checkout", "-B", branch, f"origin/{branch}"], repo_dir, repo_ref.clone_url, token)
        _git(["pull", "--ff-only"], repo_dir, repo_ref.clone_url, token, check=False)

    head = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir, repo_ref.clone_url, token).stdout.strip()
    commit = _git(["rev-parse", "HEAD"], repo_dir, repo_ref.clone_url, token).stdout.strip()
    return {
        "action": action,
        "project": repo_ref.project,
        "repo": repo_ref.repo,
        "path": str(repo_dir),
        "branch": head,
        "commit": commit,
    }


def _git(
    args: list[str],
    cwd: Path | None,
    clone_url: str,
    token: str,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = f"http.{_git_config_url_scope(clone_url)}.extraHeader"
    env["GIT_CONFIG_VALUE_0"] = f"Authorization: Bearer {token}"
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"git {' '.join(args)} failed")
    return result


def _git_config_url_scope(clone_url: str) -> str:
    parsed = urlparse(clone_url)
    return f"{parsed.scheme}://{parsed.netloc}/"


def _validate_key(label: str, value: str) -> None:
    if not value or not SAFE_KEY_RE.match(value):
        raise ValueError(f"Invalid {label}: {value!r}")


def _validate_branch(value: str) -> None:
    if not value or value.startswith("-") or any(part in value for part in ("..", "~", "^", ":", "\\", " ")):
        raise ValueError(f"Invalid branch name: {value!r}")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clone or update a Bitbucket Server repository for Hermes.")
    parser.add_argument("repo_url", help="Bitbucket Server web URL or clone URL")
    parser.add_argument("--branch", "-b", default="", help="Optional branch to checkout")
    args = parser.parse_args(argv)

    server_url = os.environ.get("BITBUCKET_SERVER_URL", "https://bitbucket.glb.syfbank.com")
    token = os.environ.get("BITBUCKET_SERVER_BEARER_TOKEN", "")
    workdir = Path(os.environ.get("BITBUCKET_WORKDIR", "workspace/repos"))
    allowed_projects = _csv(os.environ.get("BITBUCKET_ALLOWED_PROJECTS", "EUI"))

    try:
        repo_ref = parse_repo_url(args.repo_url, server_url, allowed_projects)
        result = clone_or_update(repo_ref, workdir, token, args.branch)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1

    print(json.dumps({"ok": True, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import pytest

from hermes_pcf.bitbucket_pr import PullRequestRef, parse_pull_request_url, summarize_pull_request


def test_parse_bitbucket_pull_request_web_url() -> None:
    ref = parse_pull_request_url(
        "https://bitbucket.glb.syfbank.com/projects/EUI/repos/vista/pull-requests/2331/overview",
        "https://bitbucket.glb.syfbank.com",
        ["EUI"],
    )

    assert ref.project == "EUI"
    assert ref.repo == "vista"
    assert ref.pull_request_id == 2331
    assert ref.api_url == (
        "https://bitbucket.glb.syfbank.com/rest/api/1.0/projects/EUI/repos/vista"
        "/pull-requests/2331"
    )


def test_parse_bitbucket_pull_request_rest_url() -> None:
    ref = parse_pull_request_url(
        "https://bitbucket.glb.syfbank.com/rest/api/1.0/projects/EUI/repos/vista/pull-requests/2331",
        "https://bitbucket.glb.syfbank.com",
        ["EUI"],
    )

    assert ref.project == "EUI"
    assert ref.repo == "vista"
    assert ref.pull_request_id == 2331


def test_parse_bitbucket_pull_request_rejects_disallowed_project() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        parse_pull_request_url(
            "https://bitbucket.glb.syfbank.com/projects/ABC/repos/vista/pull-requests/2331/overview",
            "https://bitbucket.glb.syfbank.com",
            ["EUI"],
        )


def test_summarize_pull_request_keeps_description_and_refs() -> None:
    summary = summarize_pull_request(
        PullRequestRef(
            project="EUI",
            repo="vista",
            pull_request_id=2331,
            api_url="https://bitbucket/rest/api/1.0/projects/EUI/repos/vista/pull-requests/2331",
        ),
        {
            "title": "Add routing",
            "description": "This PR updates react-router usage.",
            "state": "OPEN",
            "fromRef": {"displayId": "feature/router", "latestCommit": "abc123"},
            "toRef": {"displayId": "develop", "latestCommit": "def456"},
            "author": {"user": {"displayName": "A User"}, "role": "AUTHOR"},
            "reviewers": [{"user": {"displayName": "Reviewer"}, "status": "UNAPPROVED"}],
        },
    )

    assert summary["title"] == "Add routing"
    assert summary["description"] == "This PR updates react-router usage."
    assert summary["from_ref"]["display_id"] == "feature/router"
    assert summary["to_ref"]["display_id"] == "develop"
    assert summary["author"]["display_name"] == "A User"
    assert summary["reviewers"][0]["display_name"] == "Reviewer"

from __future__ import annotations

import pytest

from hermes_pcf.bitbucket_clone import parse_repo_url


def test_parse_bitbucket_web_url_to_clone_url() -> None:
    ref = parse_repo_url(
        "https://bitbucket.glb.syfbank.com/projects/EUI/repos/vista/",
        "https://bitbucket.glb.syfbank.com",
        ["EUI"],
    )

    assert ref.project == "EUI"
    assert ref.repo == "vista"
    assert ref.clone_url == "https://bitbucket.glb.syfbank.com/scm/eui/vista.git"


def test_parse_bitbucket_clone_url() -> None:
    ref = parse_repo_url(
        "https://bitbucket.glb.syfbank.com/scm/eui/vista.git",
        "https://bitbucket.glb.syfbank.com",
        ["EUI"],
    )

    assert ref.project == "EUI"
    assert ref.repo == "vista"


def test_rejects_unallowed_project() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        parse_repo_url(
            "https://bitbucket.glb.syfbank.com/projects/ABC/repos/vista/",
            "https://bitbucket.glb.syfbank.com",
            ["EUI"],
        )


def test_rejects_unexpected_host() -> None:
    with pytest.raises(ValueError, match="Repository host"):
        parse_repo_url(
            "https://example.com/projects/EUI/repos/vista/",
            "https://bitbucket.glb.syfbank.com",
            ["EUI"],
        )

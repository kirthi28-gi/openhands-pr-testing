import json
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / ".github/scripts"))

import prepare_regression_tests as regression  # noqa: E402


SALEOR_REPO = "saleor/saleor"
CONTENT = "def test_saleor_regression(): pass\n"


def test_namespace_resolution_preserves_legacy_and_adds_saleor():
    legacy = regression.namespace_for(None)
    assert legacy["repository"] == regression.EXPECTED_REPOSITORY
    assert legacy["archive_namespace"] == "openhands-pr-testing"

    saleor = regression.namespace_for(SALEOR_REPO)
    assert saleor["repository"] == SALEOR_REPO
    assert saleor["metadata_dir"].as_posix() == "metadata/saleor"
    assert saleor["test_root"].as_posix() == "tests/generated/saleor"
    assert saleor["archive_namespace"] == "saleor"

    with pytest.raises(RuntimeError, match="No approved regression namespace"):
        regression.namespace_for("untrusted/repo")


def test_malformed_namespace_configs_fail_closed(tmp_path):
    cases = [
        [],
        {"version": 2, "repositories": {}},
        {"version": 1, "repositories": []},
        {"version": 1, "repositories": {"a/b": {"metadata_dir": "../etc"}}},
        {"version": 1, "repositories": {
            "a/b": {
                "metadata_dir": "metadata/a",
                "test_root": "tests/generated/a",
                "archive_namespace": "../escape",
            }
        }},
    ]
    for index, content in enumerate(cases):
        path = tmp_path / f"config-{index}.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        with pytest.raises(RuntimeError):
            regression.load_namespaces(path)


def _write_saleor_approval(repo, pr_number=7, source_repository=SALEOR_REPO):
    relative = f"tests/generated/saleor/PR-{pr_number}/test_regression.py"
    test = repo / relative
    test.parent.mkdir(parents=True, exist_ok=True)
    test.write_text(CONTENT, encoding="utf-8")

    metadata_dir = repo / "metadata/saleor"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / f"PR-{pr_number}.json").write_text(json.dumps({
        "validation_status": "passed",
        "approval_status": "approved",
        "source_repository": source_repository,
        "source_pr_number": pr_number,
        "test_files": [relative],
    }), encoding="utf-8")
    return relative


def test_prepare_packages_saleor_namespace(tmp_path):
    repo = tmp_path / "regression"
    _write_saleor_approval(repo)
    output = tmp_path / "saleor.tar.gz"

    regression.prepare(repo, output, SALEOR_REPO)

    expected = "tests/regression/saleor/PR-7/test_regression.py"
    with tarfile.open(output, "r:gz") as archive:
        assert expected in archive.getnames()
        assert archive.extractfile(expected).read().decode() == CONTENT


def test_cross_repository_metadata_is_rejected(tmp_path):
    repo = tmp_path / "regression"
    _write_saleor_approval(repo, source_repository=regression.EXPECTED_REPOSITORY)
    with pytest.raises(RuntimeError, match="Unexpected source repository"):
        regression.prepare(repo, tmp_path / "out.tar.gz", SALEOR_REPO)


def test_empty_saleor_namespace_packs_no_tests(tmp_path):
    """An initialized but unapproved namespace must not publish any tests."""
    repo = tmp_path / "regression"
    (repo / "metadata/saleor").mkdir(parents=True)
    (repo / "tests/generated/saleor").mkdir(parents=True)
    output = tmp_path / "empty-saleor.tar.gz"

    regression.prepare(repo, output, SALEOR_REPO)

    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
        assert names == ["tests/regression"]
        assert not any(name.startswith("tests/regression/saleor/") for name in names)


def test_legacy_namespace_still_accepts_no_approved_tests(tmp_path):
    repo = tmp_path / "regression"
    (repo / regression.METADATA_DIR).mkdir(parents=True)
    output = tmp_path / "empty-legacy.tar.gz"

    regression.prepare(repo, output)

    with tarfile.open(output, "r:gz") as archive:
        assert archive.getnames() == ["tests/regression"]

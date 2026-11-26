import importlib.util
import os
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github/workflows/openhands-test-generation.yml"
data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

spec = importlib.util.spec_from_file_location(
    "validate_target_repository",
    ROOT / ".github/scripts/validate_target_repository.py",
)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def steps(job):
    return data["jobs"][job]["steps"]


def step_by_name(job, name):
    return next(step for step in steps(job) if step.get("name") == name)


def test_workflow_keeps_both_triggers_and_dispatch_inputs():
    triggers = data[True] if True in data else data.get("on")
    assert "pull_request_target" in triggers
    assert triggers["pull_request_target"]["branches"] == ["main"]
    assert "workflow_dispatch" in triggers
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert set(inputs) >= {
        "target_repository", "base_sha", "head_sha", "pr_number"
    }


def test_trusted_checkouts_use_event_or_validated_dispatch_inputs():
    generate_checkout = step_by_name("generate-tests", "Checkout PR source without credentials")
    assert "inputs.head_sha" in generate_checkout["with"]["ref"]
    assert "inputs.target_repository" in generate_checkout["with"]["repository"]
    assert generate_checkout["with"]["persist-credentials"] is False

    for job in ("generate-tests", "publish-results"):
        automation = step_by_name(job, "Checkout trusted base branch automation")
        assert "github.sha" in automation["with"]["ref"]
        assert "Validate workflow inputs" in [step.get("name") for step in steps(job)]


def test_generate_step_never_receives_a_github_write_token():
    generate = step_by_name("generate-tests", "Generate and execute tests in OpenHands sandbox")
    env = generate["env"]
    assert "GH_TOKEN" not in env
    assert "GITHUB_TOKEN" not in env
    assert env["TARGET_REPOSITORY"] == "${{ inputs.target_repository || github.repository }}"


def test_regression_preparation_uses_repository_namespace():
    prepare = step_by_name("generate-tests", "Prepare approved regression tests")
    assert "--source-repository" in prepare["run"]
    assert "TARGET_REPOSITORY" in prepare["env"]


def test_publisher_job_keeps_separate_permissions_and_trusted_code():
    publisher = data["jobs"]["publish-results"]
    assert publisher["needs"] == "generate-tests"
    assert publisher["permissions"] == {
        "contents": "read", "pull-requests": "write"
    }
    publish_step = step_by_name("publish-results", "Publish validated results to the PR")
    assert "trusted/.github/scripts/publish_test_results.py" in publish_step["run"]
    assert "--repository \"$TARGET_REPOSITORY\"" in publish_step["run"]
    assert "GITHUB_REPOSITORY}" not in publish_step["run"]
    for step in steps("publish-results"):
        run = step.get("run", "")
        assert "pr-source" not in run
        assert "python -c" not in run


def test_diff_fetch_uses_target_repository():
    diff = step_by_name("generate-tests", "Prepare PR diff")
    assert "https://github.com/${TARGET_REPOSITORY}.git" in diff["run"]
    publisher_diff = step_by_name("publish-results", "Regenerate trusted PR diff")
    assert "https://github.com/${TARGET_REPOSITORY}.git" in publisher_diff["run"]


def test_input_validation_allows_only_configured_repositories(monkeypatch):
    base = "a" * 40
    head = "b" * 40
    monkeypatch.setenv("GITHUB_REPOSITORY", "kirthi28-gi/openhands-pr-testing")
    assert validator.validate_inputs(
        "saleor/saleor", base, head, "42", ROOT
    ) == {"repository": "saleor/saleor", "pr_number": 42}
    with pytest.raises(ValueError, match="not approved"):
        validator.validate_inputs("evil/repo", base, head, "42", ROOT)


def test_input_validation_rejects_malformed_shas_and_numbers(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "kirthi28-gi/openhands-pr-testing")
    for args in (
        ("saleor/saleor", "short", "b" * 40, "42"),
        ("saleor/saleor", "a" * 40, "../etc", "42"),
        ("saleor/saleor", "a" * 40, "b" * 40, "true"),
        ("saleor/saleor", "a" * 40, "b" * 40, "0"),
    ):
        with pytest.raises(ValueError):
            validator.validate_inputs(*args, ROOT)


def test_publisher_rejects_report_for_another_repository():
    from test_publish_github import publish, publisher as pub, sample_report

    report = sample_report(target_repository="other/repo")
    with pytest.raises(pub.PublisherError, match="target repository"):
        publish(report=report)

"""Focused tests for Fix 4 reporting categories."""

import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

import generate_tests as gt  # noqa: E402
from pr_test_harness import (  # noqa: E402
    _diagnosis_was_rejected,
    ensure_failure_artifacts,
    final_failure_category,
    workflow_failure_category,
)


def _attempt(category=None, exit_code=1, **extra):
    attempt = {"attempt": 1, "exit_code": exit_code}
    if category is not None:
        attempt["diagnosis"] = {
            "validated": True,
            "category": category,
            "decision_reason": "Diagnosis has exact output evidence and a root cause.",
        }
    attempt.update(extra)
    return attempt


def _report(failures=None, **extra):
    report = {
        "failures": failures or [],
        "dependency_installation_failures": [],
        "repair_history": [],
    }
    report.update(extra)
    return report


def test_healthy_environment_with_rejected_diagnosis_uses_diagnosis_rejected():
    rejection = "Diagnosis rejected: evidence mismatch; human review required."
    report = _report(
        failures=[{"attempt": 1, "category": "application bug"}],
        repair_history=[{
            "attempt": 1,
            "decision": "human_review",
            "decision_reason": rejection,
        }],
    )
    attempts = [_attempt(exit_code=1, repair_decision_reason=rejection)]

    assert final_failure_category(report, attempts, True) == "diagnosis_rejected"
    assert workflow_failure_category(report, True, attempts) == "diagnosis_rejected"
    assert gt.report_failure_categories(report, attempts, True) == (
        "diagnosis_rejected",
        "diagnosis_rejected",
    )


def test_failed_environment_check_uses_environment():
    report = _report()

    assert final_failure_category(report, [], False) == "environment"
    assert workflow_failure_category(report, False, []) == "environment"


def test_missing_diagnosis_with_ready_environment_uses_unknown():
    report = _report()
    attempts = [_attempt(exit_code=1)]

    assert final_failure_category(report, attempts, True) == "unknown"
    assert workflow_failure_category(report, True, attempts) == "unknown"


def test_application_bug_classification_is_preserved():
    report = _report(failures=[{"attempt": 1, "category": "application bug"}])
    attempts = [_attempt("application bug")]

    assert final_failure_category(report, attempts, True) == "application bug"
    assert workflow_failure_category(report, True, attempts) == "application bug"


def test_regression_classification_is_preserved():
    report = _report(failures=[{"attempt": 1, "category": "regression"}])
    attempts = [_attempt("regression")]

    assert final_failure_category(report, attempts, True) == "regression"
    assert workflow_failure_category(report, True, attempts) == "regression"


def test_build_classification_is_preserved():
    report = _report(failures=[{"attempt": 1, "category": "build"}])
    attempts = [_attempt("build")]

    assert final_failure_category(report, attempts, True) == "build"
    assert workflow_failure_category(report, True, attempts) == "build"


def test_dependency_classification_is_preserved_when_environment_fails():
    report = _report(dependency_installation_failures=[{
        "name": "pkg", "manager": "pip", "command": "pip install pkg",
        "exit_code": 1, "output": "failed",
    }])

    assert final_failure_category(report, [], False) == "dependency"
    assert workflow_failure_category(report, False, []) == "dependency"


def test_fallback_report_uses_rejected_diagnosis_category(tmp_path):
    rejection = "Diagnosis rejected: evidence mismatch; human review required."
    report_path = tmp_path / "openhands-report.json"
    report_path.write_text(json.dumps({
        "result": "failed",
        "report_valid": False,
        "environment_ready": True,
        "environment": {"status": "ready"},
        "failures": [],
        "dependency_installation_failures": [],
        "harness_attempts": [{
            "attempt": 1,
            "exit_code": 1,
            "repair_decision": "human_review",
            "repair_decision_reason": rejection,
        }],
        "repair_history": [{
            "attempt": 1,
            "decision": "human_review",
            "decision_reason": rejection,
        }],
    }), encoding="utf-8")

    ensure_failure_artifacts(tmp_path, "RuntimeError", "traceback")

    report = json.loads(report_path.read_text())
    assert report["final_failure_category"] == "diagnosis_rejected"
    assert report["workflow_failure_category"] == "diagnosis_rejected"


def test_fallback_report_uses_environment_when_environment_failed(tmp_path):
    report_path = tmp_path / "openhands-report.json"
    report_path.write_text(json.dumps({
        "result": "failed",
        "report_valid": False,
        "environment_ready": False,
        "environment": {"status": "blocked"},
        "failures": [],
        "dependency_installation_failures": [],
        "harness_attempts": [],
        "repair_history": [],
    }), encoding="utf-8")

    ensure_failure_artifacts(tmp_path, "RuntimeError", "traceback")

    report = json.loads(report_path.read_text())
    assert report["final_failure_category"] == "environment"
    assert report["workflow_failure_category"] == "environment"


def test_rejection_is_found_when_first_field_is_unrelated():
    report = {
        "repair_history": [{
            "attempt": 1,
            "repair_decision_reason": (
                "Test integrity guard failed before quarantined rerun."
            ),
            "diagnosis_validation_error": (
                "Diagnosis rejected: evidence mismatch; human review required."
            ),
        }]
    }

    assert _diagnosis_was_rejected(report, []) is True
    assert final_failure_category(report, [], True) == "diagnosis_rejected"


def test_normal_summary_written_to_step_summary_matches_json_categories(tmp_path):
    report = {
        "final_failure_category": "diagnosis_rejected",
        "workflow_failure_category": "diagnosis_rejected",
        "report_valid": False,
    }
    report_path = tmp_path / "openhands-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    summary_path = tmp_path / "summary.md"
    result = SimpleNamespace(exit_code=1, stdout="FAILED test_x", stderr="")

    text = gt.normal_summary_text(
        "19", "base", "head", "image", result, [{"attempt": 1}], loaded
    )
    gt.write_summary_text(str(summary_path), text)

    summary = summary_path.read_text(encoding="utf-8")
    assert summary == text
    assert (
        f"- Failure category: {loaded['final_failure_category']}"
    ) in summary
    assert (
        f"- Workflow failure category: {loaded['workflow_failure_category']}"
    ) in summary


def test_fallback_summary_written_to_step_summary_matches_json_categories(
    tmp_path, monkeypatch
):
    automation = tmp_path / "trusted"
    output_dir = automation / "generated-tests"
    output_dir.mkdir(parents=True)
    report_path = output_dir / "openhands-report.json"
    report_path.write_text(json.dumps({
        "result": "failed",
        "report_valid": False,
        "environment_ready": True,
        "environment": {"status": "ready"},
        "failures": [],
        "dependency_installation_failures": [],
        "harness_attempts": [{
            "attempt": 1,
            "exit_code": 1,
            "repair_decision_reason": (
                "Diagnosis rejected: evidence mismatch; human review required."
            ),
        }],
        "repair_history": [],
        "modified_files": [],
        "proposed_patches": [],
        "verified_test_results": [],
        "blockers": [],
    }), encoding="utf-8")
    summary_path = tmp_path / "summary.md"
    monkeypatch.setenv("AUTOMATION_DIR", str(automation))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))

    def raising():
        raise RuntimeError("boom")

    monkeypatch.setattr(gt, "_main", raising)
    with pytest.raises(RuntimeError, match="boom"):
        gt.main()

    fallback = json.loads(report_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")
    assert fallback["final_failure_category"] == "diagnosis_rejected"
    assert fallback["workflow_failure_category"] == "diagnosis_rejected"
    assert (
        f"- Failure category: {fallback['final_failure_category']}"
    ) in summary
    assert (
        f"- Workflow failure category: {fallback['workflow_failure_category']}"
    ) in summary


def test_environment_failure_precedence_over_rejected_diagnosis():
    rejection = "Diagnosis rejected: evidence mismatch; human review required."
    report = {
        "failures": [],
        "dependency_installation_failures": [],
        "repair_history": [{
            "attempt": 1,
            "decision": "human_review",
            "decision_reason": rejection,
        }],
    }
    attempts = [{
        "attempt": 1,
        "exit_code": 1,
        "diagnosis": {"validated": False, "decision_reason": rejection},
    }]

    assert final_failure_category(report, attempts, False) == "environment"
    assert workflow_failure_category(report, False, attempts) == "environment"


def test_environment_failure_precedence_keeps_dependency_category():
    rejection = "Diagnosis rejected: evidence mismatch; human review required."
    report = {
        "failures": [],
        "dependency_installation_failures": [{
            "name": "pkg", "manager": "pip", "command": "pip install pkg",
            "exit_code": 1, "output": "failed",
        }],
        "repair_history": [{
            "attempt": 1,
            "decision": "human_review",
            "decision_reason": rejection,
        }],
    }
    attempts = [{"attempt": 1, "exit_code": 1}]

    assert final_failure_category(report, attempts, False) == "dependency"
    assert workflow_failure_category(report, False, attempts) == "dependency"


def test_nested_diagnosis_fields_are_inspected_consistently():
    report = {
        "repair_history": [{
            "attempt": 1,
            "diagnosis": {
                "validated": False,
                "decision_reason": (
                    "No valid structured diagnosis was available; "
                    "human review required."
                ),
                "error": "Diagnosis rejected: evidence mismatch",
            },
        }]
    }

    assert _diagnosis_was_rejected(report, []) is True
    assert final_failure_category(report, [], True) == "diagnosis_rejected"


def test_unrelated_nested_text_is_not_rejection():
    report = {
        "repair_history": [{
            "attempt": 1,
            "diagnosis": {
                "validated": False,
                "decision_reason": (
                    "No valid structured diagnosis was available; "
                    "human review required."
                ),
                "note": "an unrelated note",
            },
        }]
    }

    assert _diagnosis_was_rejected(report, []) is False
    assert final_failure_category(report, [], True) == "unknown"


def test_normal_execution_path_writes_categories_to_summary(tmp_path, monkeypatch):
    automation = tmp_path / "trusted"
    source = tmp_path / "source"
    automation.mkdir()
    source.mkdir()
    regression_input = automation / "regression-input"
    regression_input.mkdir()
    corpus_archive = regression_input / "approved-regression-tests.tar.gz"
    with tarfile.open(corpus_archive, "w:gz") as archive:
        directory = tarfile.TarInfo("tests/regression/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        directory.size = 0
        archive.addfile(directory)
        payload = b"def test_ok(): pass\n"
        member = tarfile.TarInfo("tests/regression/PR-19/test_ok.py")
        member.type = tarfile.REGTYPE
        member.mode = 0o644
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    scripts = automation / ".github/scripts"
    scripts.mkdir(parents=True)
    (scripts / "safe_extract_regression_tests.py").write_text("# helper\n", encoding="utf-8")
    diff_file = tmp_path / "pr.diff"
    diff_file.write_text("diff", encoding="utf-8")
    summary_path = tmp_path / "summary.md"

    for name, value in {
        "AUTOMATION_DIR": str(automation),
        "OH_SANDBOX_NETWORK": "oh-pr-sandbox-test",
        "PR_SOURCE_DIR": str(source),
        "PR_DIFF_FILE": str(diff_file),
        "PR_NUMBER": "19",
        "PR_BASE_SHA": "base",
        "PR_HEAD_SHA": "head",
        "LLM_API_KEY": "test-key",
        "GITHUB_STEP_SUMMARY": str(summary_path),
    }.items():
        monkeypatch.setenv(name, value)

    agent_report = {
        "schema_version": 1,
        "detected_stack": {},
        "environment": {"status": "ready"},
        "test_commands": ["pytest -q"],
        "failures": [],
        "dependency_installation_failures": [],
        "root_causes": [],
        "modified_files": [],
        "proposed_patches": [],
        "verified_test_results": [],
        "blockers": [],
    }
    attempt_records = [{
        "attempt": 1,
        "command": "pytest -q",
        "exit_code": 1,
        "output": "FAILED tests/generated/test_x.py::test_y",
        "diagnosis": {
            "validated": False,
            "category": "unknown",
            "decision_reason": (
                "No valid structured diagnosis was available; "
                "human review required."
            ),
            "error": "Diagnosis rejected: evidence mismatch",
        },
        "repair_decision": "human_review",
    }]

    class FakeConversation:
        def send_message(self, message):
            return None

        def run(self):
            return None

        def close(self):
            return None

    class FakeWorkspace:
        def execute_command(self, command, timeout=600):
            if "wc -c < /workspace/openhands-test-command" in command:
                return SimpleNamespace(exit_code=0, stdout="8", stderr="")
            if "cat -- /workspace/openhands-test-command" in command:
                return SimpleNamespace(exit_code=0, stdout="pytest -q", stderr="")
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        def file_upload(self, source_path, destination_path):
            return SimpleNamespace(success=True, error=None)

        def file_download(self, source_path, destination_path):
            content = (
                json.dumps(agent_report)
                if "openhands-report.json" in source_path else ""
            )
            Path(destination_path).write_text(content, encoding="utf-8")
            return SimpleNamespace(success=True, error=None)

        def close(self):
            return None

    class FakeDockerWorkspace:
        def __init__(self, **kwargs):
            self.workspace = FakeWorkspace()

        def __enter__(self):
            return self.workspace

        def __exit__(self, exc_type, exc, tb):
            return False

    def unchanged(*args, **kwargs):
        return {
            "unchanged": True,
            "changed_paths": [],
            "errors": [],
            "restored": False,
        }

    monkeypatch.setattr(gt, "block_embedded_dns", lambda *a, **k: None)
    monkeypatch.setattr(gt, "get_changed_files", lambda *a, **k: ["app.py"])
    monkeypatch.setattr(gt, "LLM", lambda **k: SimpleNamespace())
    monkeypatch.setattr(gt, "get_default_agent", lambda **k: SimpleNamespace())
    monkeypatch.setattr(gt, "Conversation", lambda **k: FakeConversation())
    monkeypatch.setattr(gt, "create_source_archive", lambda *a, **k: None)
    monkeypatch.setattr(gt, "DockerWorkspace", lambda **k: FakeDockerWorkspace())
    monkeypatch.setattr(gt, "capture_generated_tests_snapshot", lambda *a, **k: {})
    monkeypatch.setattr(gt, "capture_repository_tests_snapshot", lambda *a, **k: {})
    monkeypatch.setattr(
        gt, "validate_generated_tests",
        lambda *a, **k: {
            "valid": True,
            "suspected_bugs": [],
            "classification": "passed",
            "errors": [],
        },
    )
    monkeypatch.setattr(gt, "ensure_generated_tests_unchanged", unchanged)
    monkeypatch.setattr(gt, "ensure_repository_tests_unchanged", unchanged)
    monkeypatch.setattr(gt, "validate_test_command", lambda *a, **k: "pytest -q")
    monkeypatch.setattr(
        gt, "verify_test_environment",
        lambda *a, **k: {
            "ready": True,
            "failures": [],
            "blockers": [],
            "checks": [],
            "limitations": [],
        },
    )
    monkeypatch.setattr(gt, "sandbox_test_command", lambda *a, **k: "pytest -q")
    monkeypatch.setattr(
        gt, "run_repair_loop",
        lambda *a, **k: (
            SimpleNamespace(exit_code=1, stdout="FAILED", stderr=""),
            attempt_records,
        ),
    )

    with pytest.raises(RuntimeError, match="Repository test command reported failures"):
        gt._main()

    final_report = json.loads(
        (automation / "generated-tests/openhands-report.json").read_text(encoding="utf-8")
    )
    summary = summary_path.read_text(encoding="utf-8")
    assert final_report["final_failure_category"] == "diagnosis_rejected"
    assert final_report["workflow_failure_category"] == "diagnosis_rejected"
    assert (
        f"- Failure category: {final_report['final_failure_category']}"
    ) in summary
    assert (
        f"- Workflow failure category: {final_report['workflow_failure_category']}"
    ) in summary

"""Tests for trusted, evidence-backed failure classification."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

import generate_tests as gt  # noqa: E402
import pr_test_harness as harness  # noqa: E402


READY = {"ready": True, "failures": [], "checks": []}


def _classify(**kwargs):
    defaults = {
        "environment_verification": READY,
        "attempt_records": [],
        "generated_test_validation": None,
        "report": {},
        "stdout": "",
        "stderr": "",
        "workflow_error": None,
    }
    defaults.update(kwargs)
    return harness.classify_failure(**defaults)


def test_pr_syntax_error_requires_pr_owned_file_and_syntax_evidence():
    output = (
        'File "/workspace/app.py", line 10\n'
        "x = (1 +\n"
        "SyntaxError: invalid syntax\n"
    )
    result = _classify(
        attempt_records=[{"attempt": 1, "exit_code": 1, "output": output}],
        stdout=output,
    )

    assert result["category"] == "pr_syntax_error"
    assert any("app.py" in item for item in result["evidence"])
    assert any("SyntaxError" in item for item in result["evidence"])


def test_pr_syntax_error_is_not_asserted_without_owned_file():
    output = "SyntaxError: invalid syntax\n"
    result = _classify(
        attempt_records=[{"attempt": 1, "exit_code": 1, "output": output}],
        stdout=output,
    )

    assert result["category"] == "inconclusive"


def test_pr_logic_error_requires_validated_diagnosis_evidence():
    output = (
        "FAILED tests/test_calculator.py::test_add\n"
        "E AssertionError: expected 3, got 2\n"
    )
    evidence = [output.splitlines()[1]]
    diagnosis = {
        "validated": True,
        "category": "application bug",
        "evidence": evidence,
        "root_cause_evidence": evidence,
    }
    result = _classify(
        attempt_records=[{
            "attempt": 1,
            "exit_code": 1,
            "output": output,
            "diagnosis": diagnosis,
        }],
        report={"failures": [{
            "attempt": 1,
            "category": "application bug",
            "evidence": evidence,
        }]},
        stdout=output,
    )

    assert result["category"] == "pr_logic_error"
    assert result["source"] == "validated_diagnosis"
    assert result["evidence"] == evidence


def test_pr_logic_error_is_not_asserted_from_unvalidated_report_failure():
    output = (
        "FAILED tests/test_calculator.py::test_add\n"
        "E AssertionError: expected 3, got 2\n"
    )
    result = _classify(
        attempt_records=[{"attempt": 1, "exit_code": 1, "output": output}],
        report={"failures": [{
            "attempt": 1,
            "category": "application bug",
            "evidence": [output.splitlines()[1]],
        }]},
        stdout=output,
    )

    assert result["category"] == "inconclusive"


def test_incorrect_generated_test_uses_generated_validation_evidence():
    output = (
        "FAILED tests/generated/test_add.py::test_add\n"
        "E AssertionError: expected 3, got 2\n"
    )
    result = _classify(
        attempt_records=[{"attempt": 1, "exit_code": 1, "output": output}],
        generated_test_validation={
            "classification": "invalid_generated_test",
            "results": [{
                "attempt": 1,
                "exit_code": 1,
                "classification": "invalid_generated_test",
                "output": output,
            }],
        },
        stdout=output,
    )

    assert result["category"] == "incorrect_generated_test"
    assert any("tests/generated" in item for item in result["evidence"])


def test_invalid_test_setup_uses_setup_evidence():
    output = (
        "ERROR at setup of test_add\n"
        "E fixture 'mock_client' not found\n"
    )
    result = _classify(
        attempt_records=[{"attempt": 1, "exit_code": 1, "output": output}],
        generated_test_validation={
            "classification": "invalid_generated_test",
            "results": [{
                "attempt": 1,
                "exit_code": 1,
                "classification": "invalid_generated_test",
                "output": output,
            }],
        },
        stdout=output,
    )

    assert result["category"] == "invalid_test_setup"
    assert any("setup" in item.lower() for item in result["evidence"])


def test_missing_dependency_is_distinguished_from_application_defect():
    output = "ModuleNotFoundError: No module named 'litellm'\n"
    env = {
        "ready": False,
        "failures": [output.strip()],
        "checks": [{"name": "python_trusted_dependency_imports", "output": output}],
    }
    result = _classify(
        environment_verification=env,
        stdout=output,
    )

    assert result["category"] == "missing_dependency"
    assert any("litellm" in item for item in result["evidence"])


def test_sandbox_or_infrastructure_failure_uses_environment_evidence():
    output = "Could not connect to the Docker daemon: permission denied\n"
    env = {"ready": False, "failures": [output.strip()], "checks": []}
    result = _classify(environment_verification=env, stdout=output)

    assert result["category"] == "sandbox_or_infrastructure"
    assert any("Docker daemon" in item for item in result["evidence"])


def test_conflicting_dependency_and_application_evidence_is_inconclusive():
    output = (
        "FAILED tests/test_calculator.py::test_add\n"
        "ModuleNotFoundError: No module named 'litellm'\n"
    )
    diagnosis = {
        "validated": True,
        "category": "application bug",
        "evidence": [output.splitlines()[0]],
    }
    result = _classify(
        attempt_records=[{
            "attempt": 1,
            "exit_code": 1,
            "output": output,
            "diagnosis": diagnosis,
        }],
        stdout=output,
    )

    assert result["category"] == "inconclusive"
    assert result["conflicts"]


def test_conflicting_generated_setup_and_assertion_evidence_is_inconclusive():
    output = (
        "ERROR at setup of test_add\n"
        "FAILED tests/generated/test_add.py::test_add\n"
        "E AssertionError: expected 3, got 2\n"
    )
    result = _classify(
        generated_test_validation={
            "classification": "invalid_generated_test",
            "results": [{
                "attempt": 1,
                "exit_code": 1,
                "classification": "invalid_generated_test",
                "output": output,
            }],
        },
        stdout=output,
    )

    assert result["category"] == "inconclusive"
    assert any("setup" in conflict for conflict in result["conflicts"])


def test_insufficient_evidence_is_inconclusive():
    output = "unexpected failure\n"
    result = _classify(
        attempt_records=[{"attempt": 1, "exit_code": 1, "output": output}],
        stdout=output,
    )

    assert result["category"] == "inconclusive"


def test_verified_success_classifies_as_none():
    result = _classify(
        attempt_records=[{"attempt": 1, "exit_code": 0, "output": "1 passed"}],
        stdout="1 passed",
    )

    assert result["category"] == "none"


def test_classification_is_propagated_to_summaries(tmp_path):
    classification = {
        "version": 1,
        "category": "missing_dependency",
        "source": "environment_verification",
        "detail": "ModuleNotFoundError captured",
        "evidence": ["ModuleNotFoundError: No module named 'litellm'"],
        "conflicts": [],
    }
    report = {
        "final_failure_category": "dependency",
        "workflow_failure_category": "dependency",
        "report_valid": False,
        "failure_classification": classification,
    }
    result = SimpleNamespace(exit_code=1, stdout="FAILED", stderr="")

    normal = gt.normal_summary_text("20", "base", "head", "image", result, [], report)
    assert "- Failure class: missing_dependency" in normal
    assert "ModuleNotFoundError: No module named 'litellm'" in normal

    fallback = gt.fallback_summary_text(
        "RuntimeError", "dependency", "dependency", classification
    )
    assert "- Failure class: missing_dependency" in fallback
    assert "ModuleNotFoundError: No module named 'litellm'" in fallback


def test_classification_is_propagated_to_fallback_report(tmp_path):
    report_path = tmp_path / "openhands-report.json"
    report_path.write_text(json.dumps({
        "result": "failed",
        "report_valid": False,
        "environment_ready": False,
        "environment": {"status": "blocked"},
        "environment_verification": {
            "ready": False,
            "failures": ["ModuleNotFoundError: No module named 'pydantic'"],
            "checks": [],
        },
        "failures": [],
        "dependency_installation_failures": [],
        "harness_attempts": [],
        "repair_history": [],
    }), encoding="utf-8")

    harness.ensure_failure_artifacts(
        tmp_path, "RuntimeError", "ModuleNotFoundError: No module named 'pydantic'"
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["failure_classification"]["category"] == "missing_dependency"


def test_dependency_like_assertion_text_is_not_missing_dependency():
    output = (
        "FAILED tests/test_calculator.py::test_add\n"
        "E AssertionError: expected 'No module named foo', got 'real value'\n"
    )
    result = _classify(
        attempt_records=[{"attempt": 1, "exit_code": 1, "output": output}],
        stdout=output,
    )

    assert result["category"] != "missing_dependency"
    assert result["category"] == "inconclusive"


def test_application_suite_assertion_does_not_prove_generated_test_defect():
    generated_output = "generated tests failed within the retry limit\n"
    final_output = (
        "FAILED tests/test_calculator.py::test_add\n"
        "E AssertionError: expected 3, got 2\n"
    )
    result = _classify(
        attempt_records=[{"attempt": 1, "exit_code": 1, "output": final_output}],
        generated_test_validation={
            "classification": "invalid_generated_test",
            "results": [{
                "attempt": 1,
                "exit_code": 1,
                "classification": "invalid_generated_test",
                "output": generated_output,
            }],
        },
        stdout=final_output,
    )

    assert result["category"] == "inconclusive"
    assert result["source"] == "generated_test_validation"
    assert result["evidence_status"] == "unavailable"


def test_application_suite_setup_error_does_not_prove_generated_test_setup_defect():
    generated_output = "generated tests failed within the retry limit\n"
    final_output = (
        "ERROR at setup of test_add\n"
        "E fixture 'database' not found\n"
    )
    result = _classify(
        attempt_records=[{"attempt": 1, "exit_code": 1, "output": final_output}],
        generated_test_validation={
            "classification": "invalid_generated_test",
            "results": [{
                "attempt": 1,
                "exit_code": 1,
                "classification": "invalid_generated_test",
                "output": generated_output,
            }],
        },
        stdout=final_output,
    )

    assert result["category"] == "inconclusive"
    assert result["category"] != "invalid_test_setup"


def test_relative_pr_owned_file_path_supports_syntax_error():
    output = (
        'File "Calculator.py", line 10\n'
        "x = (1 +\n"
        "SyntaxError: invalid syntax\n"
    )
    result = _classify(
        attempt_records=[{"attempt": 1, "exit_code": 1, "output": output}],
        stdout=output,
    )

    assert result["category"] == "pr_syntax_error"
    assert any("Calculator.py" in item for item in result["evidence"])


@pytest.mark.parametrize(
    "path",
    [
        "tests/generated/test_x.py",
        ".github/scripts/generate_tests.py",
        "/usr/lib/python3.12/site-packages/pkg.py",
        "/tmp/outside.py",
        "../outside.py",
    ],
)
def test_excluded_and_unrelated_syntax_error_paths_are_not_pr_owned(path):
    output = (
        f'File "{path}", line 10\n'
        "SyntaxError: invalid syntax\n"
    )
    result = _classify(
        attempt_records=[{"attempt": 1, "exit_code": 1, "output": output}],
        stdout=output,
    )

    assert result["category"] == "inconclusive"


def test_inconclusive_insufficient_evidence_is_marked_unavailable():
    result = _classify(
        attempt_records=[{"attempt": 1, "exit_code": 1, "output": "unexpected failure\n"}],
        stdout="unexpected failure\n",
    )

    assert result["category"] == "inconclusive"
    assert result["evidence"] == []
    assert result["evidence_status"] == "unavailable"


def test_inconclusive_conflicting_evidence_is_marked_captured():
    output = (
        "FAILED tests/test_calculator.py::test_add\n"
        "ModuleNotFoundError: No module named 'litellm'\n"
    )
    diagnosis = {
        "validated": True,
        "category": "application bug",
        "evidence": [output.splitlines()[0]],
    }
    result = _classify(
        attempt_records=[{
            "attempt": 1,
            "exit_code": 1,
            "output": output,
            "diagnosis": diagnosis,
        }],
        stdout=output,
    )

    assert result["category"] == "inconclusive"
    assert result["evidence"]
    assert result["evidence_status"] == "captured"


def test_early_docker_failure_fallback_uses_report_readiness(tmp_path):
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

    harness.ensure_failure_artifacts(
        tmp_path, "RuntimeError", "Docker is not available: permission denied"
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    classification = report["failure_classification"]
    assert classification["category"] == "sandbox_or_infrastructure"
    assert any("permission denied" in item for item in classification["evidence"])


def test_blocked_environment_fallback_uses_workflow_error(tmp_path):
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

    harness.ensure_failure_artifacts(tmp_path, "DockerStartupError")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    classification = report["failure_classification"]
    assert classification["category"] == "sandbox_or_infrastructure"
    assert "DockerStartupError" in classification["evidence"]


def test_captured_streams_from_test_results_are_used_in_fallback(tmp_path):
    report_path = tmp_path / "openhands-report.json"
    report_path.write_text(json.dumps({
        "result": "failed",
        "report_valid": False,
        "environment_ready": True,
        "environment": {"status": "ready"},
        "failures": [],
        "dependency_installation_failures": [],
        "harness_attempts": [],
        "repair_history": [],
    }), encoding="utf-8")
    (tmp_path / "openhands-test-results.json").write_text(json.dumps({
        "result": "failed",
        "final_exit_code": 1,
        "validated_test_command": "/workspace/.venv/bin/python -m pytest -q",
        "environment_ready": True,
        "attempts": [],
        "final_stdout": "ModuleNotFoundError: No module named 'pydantic'\n",
        "final_stderr": "",
    }), encoding="utf-8")

    harness.ensure_failure_artifacts(tmp_path, "RuntimeError")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    classification = report["failure_classification"]
    assert classification["category"] == "missing_dependency"
    assert any("pydantic" in item for item in classification["evidence"])


def _write_fallback_report(tmp_path, **overrides):
    report = {
        "result": "failed",
        "report_valid": False,
        "environment_ready": False,
        "environment": {"status": "blocked"},
        "environment_verification": {
            "ready": False,
            "failures": ["ModuleNotFoundError: No module named 'pydantic'\n"],
            "checks": [],
        },
        "failures": [],
        "dependency_installation_failures": [],
        "harness_attempts": [],
        "repair_history": [],
    }
    report.update(overrides)
    (tmp_path / "openhands-report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return report


def test_fallback_test_results_matches_normal_artifact_fields(tmp_path):
    attempt = {
        "attempt": 1,
        "exit_code": 1,
        "output": "FAILED tests/generated/test_x.py::test_x\nE AssertionError: bad\n",
        "quarantined_test": "tests/generated/test_x.py::test_x",
        "quarantine_reason": "invalid generated test",
        "quarantine_evidence": ["E AssertionError: bad"],
        "quarantine_status": "pending_rerun",
    }
    _write_fallback_report(tmp_path, harness_attempts=[attempt])

    harness.ensure_failure_artifacts(tmp_path, "RuntimeError", "traceback\n")

    report = json.loads((tmp_path / "openhands-report.json").read_text(encoding="utf-8"))
    results = json.loads(
        (tmp_path / "openhands-test-results.json").read_text(encoding="utf-8")
    )
    for key in (
        "result", "final_exit_code", "validated_test_command",
        "environment_ready", "quarantined_tests", "failure_classification",
        "attempts", "final_stdout", "final_stderr",
    ):
        assert key in results
    assert results["failure_classification"] == report["failure_classification"]
    assert results["failure_classification"]["category"] == "missing_dependency"
    assert results["attempts"] == report["harness_attempts"]
    assert results["quarantined_tests"] == gt.quarantined_tests_from_attempts([attempt])
    assert isinstance(results["final_stdout"], str)
    assert isinstance(results["final_stderr"], str)


def test_fallback_preserves_and_stabilizes_captured_streams(tmp_path):
    stdout = (
        "FAILED tests/test_x.py::test_x\n"
        "ModuleNotFoundError: No module named 'pydantic'\n"
    )
    stderr = "captured stderr line\n"
    _write_fallback_report(tmp_path)
    (tmp_path / "openhands-test-results.json").write_text(json.dumps({
        "final_stdout": stdout,
        "final_stderr": stderr,
    }), encoding="utf-8")

    harness.ensure_failure_artifacts(tmp_path, "RuntimeError", "workflow traceback\n")
    first = json.loads(
        (tmp_path / "openhands-test-results.json").read_text(encoding="utf-8")
    )
    assert first["final_stdout"] == stdout
    assert first["final_stderr"] == stderr
    assert first["failure_classification"]["category"] == "missing_dependency"
    assert any("pydantic" in item for item in first["failure_classification"]["evidence"])

    # A second fallback must not duplicate the traceback into the retained
    # streams or lose the originally captured output.
    harness.ensure_failure_artifacts(tmp_path, "RuntimeError", "workflow traceback\n")
    second = json.loads(
        (tmp_path / "openhands-test-results.json").read_text(encoding="utf-8")
    )
    assert second["final_stdout"] == first["final_stdout"]
    assert second["final_stderr"] == first["final_stderr"]


@pytest.mark.parametrize(
    "existing",
    [
        None,
        "not json",
        "[]",
        json.dumps({"final_stdout": 7, "final_stderr": ["bad"]}),
        json.dumps({"final_stdout": None, "final_stderr": None}),
    ],
)
def test_fallback_handles_missing_or_malformed_existing_artifact(tmp_path, existing):
    _write_fallback_report(tmp_path)
    results_path = tmp_path / "openhands-test-results.json"
    if existing is not None:
        results_path.write_text(existing, encoding="utf-8")

    harness.ensure_failure_artifacts(tmp_path, "RuntimeError", "traceback\n")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert isinstance(results["final_stdout"], str)
    assert isinstance(results["final_stderr"], str)
    assert isinstance(results["failure_classification"], dict)
    assert isinstance(results["quarantined_tests"], list)


def test_fallback_report_streams_take_precedence_when_available(tmp_path):
    _write_fallback_report(
        tmp_path,
        final_stdout="report stdout\n",
        final_stderr="report stderr\n",
    )
    (tmp_path / "openhands-test-results.json").write_text(json.dumps({
        "final_stdout": "captured stdout\n",
        "final_stderr": "captured stderr\n",
    }), encoding="utf-8")

    harness.ensure_failure_artifacts(tmp_path, "RuntimeError", "traceback\n")

    results = json.loads(
        (tmp_path / "openhands-test-results.json").read_text(encoding="utf-8")
    )
    assert results["final_stdout"] == "report stdout\n"
    assert results["final_stderr"] == "report stderr\n"

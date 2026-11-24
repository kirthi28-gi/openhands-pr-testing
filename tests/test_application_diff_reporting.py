"""Regression tests for application diff capture and validation-stage reporting."""

import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

import generate_tests as gt  # noqa: E402
import pr_test_harness as harness  # noqa: E402


GENERIC_ERROR = "Agent report was not validated against captured test output."


def _passing_report(**overrides):
    report = {
        "schema_version": 1,
        "detected_stack": {"language": "python", "framework": "pytest"},
        "environment": {"status": "ready"},
        "test_commands": ["/workspace/.venv/bin/python -m pytest -q"],
        "failures": [],
        "dependency_installation_failures": [],
        "root_causes": [],
        "modified_files": ["tests/generated/test_pr20_additional.py"],
        "proposed_patches": [],
        "verified_test_results": [{
            "attempt": 1,
            "exit_code": 0,
            "output": "267 passed in 1.43s",
        }],
        "blockers": [],
        "test_integrity": {
            "unchanged": True,
            "note": "no git commits/HEAD objects; exact base/head comparison unavailable",
        },
    }
    report.update(overrides)
    return report


def _write_report(tmp_path, report):
    path = tmp_path / "openhands-report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_final_diff_capture_excludes_generated_caches_not_source():
    command = gt.application_diff_capture_command()

    assert command.startswith("diff -ruN ")
    assert "/tmp/openhands-pr-baseline /workspace" in command
    assert "test $code -le 1" in command
    for flag in ("--exclude=__pycache__", "--exclude=.pytest_cache"):
        assert flag in command

    # The integrity guard uses the same exclusions for the same reason.
    commands = []

    class Workspace:
        def execute_command(self, command, timeout=120):
            commands.append(command)
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

    result = gt.sandbox_application_unchanged(Workspace())
    assert result["unchanged"] is True
    guard_command = commands[0]
    for flag in ("--exclude=__pycache__", "--exclude=.pytest_cache"):
        assert flag in guard_command

    # Real application sources, including binary ones, are still compared.
    assert "--exclude=*.py" not in command
    assert "--exclude=*.pyc" not in command
    assert "--exclude=*.png" not in command


def test_genuine_binary_application_diff_still_raises_value_error():
    diff = (
        "diff -ruN /tmp/openhands-pr-baseline/assets/logo.png "
        "/workspace/assets/logo.png\n"
        "Binary files /tmp/openhands-pr-baseline/assets/logo.png and "
        "/workspace/assets/logo.png differ\n"
    )

    with pytest.raises(
        ValueError, match="binary application changes cannot be represented"
    ):
        harness.normalize_unified_git_patch(
            diff, "/tmp/openhands-pr-baseline", "/workspace"
        )


def test_passing_report_with_missing_git_metadata_is_accepted(tmp_path):
    attempts = [{"attempt": 1, "exit_code": 0, "output": "267 passed in 1.43s"}]
    path = _write_report(tmp_path, _passing_report())

    validated = harness.load_agent_report(path, attempts)

    assert validated["report_validation_errors"] == []
    assert validated["failures"] == []
    assert validated["root_causes"] == []
    assert validated["proposed_patches"] == []
    assert len(validated["verified_test_results"]) == 1


def test_patch_normalization_failure_marks_diff_invalid_and_fails_workflow():
    report = {"report_validation_errors": []}

    gt.finalize_report_validity(report, False)

    assert report["agent_report_valid"] is True
    assert report["application_diff_valid"] is False
    assert report["report_valid"] is False
    assert report["application_diff_errors"] == [gt.PATCH_CAPTURE_ERROR]
    assert report["report_validation_errors"] == []

    reason = gt.completion_failure_reason(report)
    assert reason
    assert "Application diff capture failed" in reason
    assert "Agent report failed schema or evidence validation." not in reason

    # A genuine schema/evidence failure keeps its own message.
    schema_invalid = {"report_validation_errors": ["missing required field: blockers"]}
    gt.finalize_report_validity(schema_invalid, True)
    assert (
        gt.completion_failure_reason(schema_invalid)
        == "Agent report failed schema or evidence validation."
    )
    assert schema_invalid["agent_report_valid"] is False
    assert schema_invalid["application_diff_valid"] is True
    assert schema_invalid["application_diff_errors"] == []
    assert schema_invalid["report_validation_errors"] == [
        "missing required field: blockers"
    ]


def test_fallback_preserves_specific_patch_capture_error(tmp_path):
    report = _passing_report(
        result="failed",
        report_valid=False,
        report_validation_errors=[],
        application_diff_valid=False,
        application_diff_errors=[gt.PATCH_CAPTURE_ERROR],
        harness_result="passed",
        harness_final_exit_code=0,
        harness_attempts=[],
        repair_history=[],
    )
    path = _write_report(tmp_path, report)

    harness.ensure_failure_artifacts(tmp_path, "RuntimeError", "traceback")

    final = json.loads(path.read_text(encoding="utf-8"))
    assert final["report_validation_errors"] == []
    assert GENERIC_ERROR not in final["report_validation_errors"]
    assert final["application_diff_errors"] == [gt.PATCH_CAPTURE_ERROR]


@pytest.mark.parametrize("value", [None, [], "bad", [123], [{}]])
def test_fallback_uses_generic_message_for_missing_or_invalid_errors(tmp_path, value):
    report = _passing_report(result="failed", report_valid=False)
    if value is not None:
        report["report_validation_errors"] = value
    path = _write_report(tmp_path, report)

    harness.ensure_failure_artifacts(tmp_path, "RuntimeError", "traceback")

    final = json.loads(path.read_text(encoding="utf-8"))
    assert final["report_validation_errors"] == [GENERIC_ERROR]


def test_fallback_summary_reports_patch_capture_failure_separately(tmp_path):
    report = _passing_report(
        result="failed",
        report_valid=False,
        report_validation_errors=[],
        application_diff_valid=False,
        application_diff_errors=[gt.PATCH_CAPTURE_ERROR],
    )
    _write_report(tmp_path, report)
    harness.ensure_failure_artifacts(tmp_path, "RuntimeError", "traceback")

    errors = gt.fallback_validation_errors(tmp_path)
    assert gt.PATCH_CAPTURE_ERROR in errors
    assert GENERIC_ERROR not in errors

    text = gt.fallback_summary_text(
        "RuntimeError", "unknown", "unknown", {}, errors
    )
    assert "- Validation errors:" in text
    assert gt.PATCH_CAPTURE_ERROR in text
    assert GENERIC_ERROR not in text


@pytest.mark.parametrize(
    "content",
    [None, "not json", "[]", json.dumps({"report_validation_errors": "bad"})],
)
def test_fallback_validation_errors_handles_missing_or_malformed_report(tmp_path, content):
    if content is not None:
        (tmp_path / "openhands-report.json").write_text(content, encoding="utf-8")

    assert gt.fallback_validation_errors(tmp_path) == []


def test_validation_marker_requires_both_stages_clean(tmp_path):
    marker = tmp_path / ".report-validated"
    clean = {
        "report_valid": True,
        "report_validation_errors": [],
        "application_diff_errors": [],
    }

    assert harness.update_validation_marker(marker, clean) is True
    assert marker.is_file()

    patch_failed = dict(clean, report_valid=False, application_diff_errors=[gt.PATCH_CAPTURE_ERROR])
    assert harness.update_validation_marker(marker, patch_failed) is False
    assert not marker.exists()

    schema_failed = dict(clean, report_valid=False, report_validation_errors=["missing field"])
    assert harness.update_validation_marker(marker, schema_failed) is False
    assert not marker.exists()


def test_normal_summary_distinguishes_agent_report_and_diff_validity():
    report = {
        "report_valid": False,
        "agent_report_valid": True,
        "application_diff_valid": False,
        "final_failure_category": None,
        "workflow_failure_category": None,
    }
    result = SimpleNamespace(exit_code=1, stdout="", stderr="")

    text = gt.normal_summary_text("20", "base", "head", "image", result, [], report)

    assert "- Agent report valid: yes" in text
    assert "- Application diff valid: no" in text


# ---- Production-path integration tests (controlled _main()/main() harness) ----


BINARY_DIFF = (
    "diff -ruN /tmp/openhands-pr-baseline/assets/logo.png "
    "/workspace/assets/logo.png\n"
    "Binary files /tmp/openhands-pr-baseline/assets/logo.png and "
    "/workspace/assets/logo.png differ\n"
)


def _run_pipeline(
    tmp_path,
    monkeypatch,
    agent_report,
    patch_text="",
    *,
    use_main=False,
):
    """Run the real _main()/main() with a recording fake sandbox."""
    automation = tmp_path / "trusted"
    source = tmp_path / "source"
    automation.mkdir()
    source.mkdir()
    scripts = automation / ".github/scripts"
    scripts.mkdir(parents=True)
    (scripts / "safe_extract.py").write_text("# trusted extractor\n", encoding="utf-8")

    regression_input = automation / "regression-input"
    regression_input.mkdir()
    with tarfile.open(
        regression_input / "approved-regression-tests.tar.gz", "w:gz"
    ) as archive:
        directory = tarfile.TarInfo("tests/regression/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        directory.size = 0
        archive.addfile(directory)
        payload = b"def test_private(): pass\n"
        member = tarfile.TarInfo("tests/regression/PR-20/test_private.py")
        member.type = tarfile.REGTYPE
        member.mode = 0o644
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    diff_file = tmp_path / "pr.diff"
    diff_file.write_text("diff", encoding="utf-8")
    summary_path = tmp_path / "summary.md"
    for name, value in {
        "AUTOMATION_DIR": str(automation),
        "OH_SANDBOX_NETWORK": "oh-pr-sandbox-test",
        "PR_SOURCE_DIR": str(source),
        "PR_DIFF_FILE": str(diff_file),
        "PR_NUMBER": "20",
        "PR_BASE_SHA": "base",
        "PR_HEAD_SHA": "head",
        "LLM_API_KEY": "llm_DO_NOT_USE_in_test",
        "GITHUB_STEP_SUMMARY": str(summary_path),
    }.items():
        monkeypatch.setenv(name, value)

    attempts = [{"attempt": 1, "exit_code": 0, "output": "267 passed"}]
    commands = []

    class Workspace:
        def execute_command(self, command, timeout=600):
            commands.append(command)
            if "wc -c < /workspace/openhands-test-command" in command:
                return SimpleNamespace(exit_code=0, stdout="8", stderr="")
            if "cat -- /workspace/openhands-test-command" in command:
                return SimpleNamespace(
                    exit_code=0,
                    stdout="/workspace/.venv/bin/python -m pytest -q",
                    stderr="",
                )
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        def file_upload(self, source_path, destination_path):
            return SimpleNamespace(success=True, error=None)

        def file_download(self, source_path, destination_path):
            if "openhands-report.json" in source_path:
                content = json.dumps(agent_report)
            elif "openhands-pr-diff" in source_path:
                content = patch_text
            else:
                content = ""
            Path(destination_path).write_text(content, encoding="utf-8")
            return SimpleNamespace(success=True, error=None)

        def close(self):
            return None

    class RecordingDockerWorkspace:
        def __init__(self, **kwargs):
            self.workspace = Workspace()

        def __enter__(self):
            return self.workspace

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConversation:
        def send_message(self, message):
            return None

        def run(self):
            return None

        def close(self):
            return None

    def unchanged(*args, **kwargs):
        return {
            "unchanged": True,
            "changed_paths": [],
            "errors": [],
            "restored": False,
        }

    monkeypatch.setattr(gt, "block_embedded_dns", lambda *a, **k: None)
    monkeypatch.setattr(gt, "install_verification_dependencies", lambda *a, **k: None)
    monkeypatch.setattr(gt, "preflight_verification_imports", lambda *a, **k: None)
    monkeypatch.setattr(gt, "get_changed_files", lambda *a, **k: ["app.py"])
    monkeypatch.setattr(gt, "LLM", lambda **k: SimpleNamespace())
    monkeypatch.setattr(gt, "get_default_agent", lambda **k: SimpleNamespace())
    monkeypatch.setattr(gt, "Conversation", lambda **k: FakeConversation())
    monkeypatch.setattr(gt, "create_source_archive", lambda *a, **k: None)
    monkeypatch.setattr(gt, "DockerWorkspace", RecordingDockerWorkspace)
    monkeypatch.setattr(gt, "capture_generated_tests_snapshot", lambda *a, **k: {})
    monkeypatch.setattr(gt, "capture_repository_tests_snapshot", lambda *a, **k: {})
    monkeypatch.setattr(
        gt,
        "validate_generated_tests",
        lambda *a, **k: {
            "valid": True,
            "suspected_bugs": [],
            "classification": "passed",
            "errors": [],
        },
    )
    monkeypatch.setattr(gt, "ensure_generated_tests_unchanged", unchanged)
    monkeypatch.setattr(gt, "ensure_repository_tests_unchanged", unchanged)
    monkeypatch.setattr(
        gt,
        "validate_test_command",
        lambda *a, **k: "/workspace/.venv/bin/python -m pytest -q",
    )
    monkeypatch.setattr(
        gt,
        "verify_test_environment",
        lambda *a, **k: {
            "ready": True,
            "failures": [],
            "blockers": [],
            "checks": [],
            "limitations": [],
        },
    )
    monkeypatch.setattr(gt, "sandbox_test_command", lambda command, **k: command)
    monkeypatch.setattr(
        gt,
        "run_repair_loop",
        lambda workspace_obj, conversation_obj, command, *a, **k: (
            SimpleNamespace(exit_code=0, stdout="267 passed", stderr=""),
            attempts,
        ),
    )

    error = None
    try:
        (gt.main if use_main else gt._main)()
    except RuntimeError as exc:
        error = exc

    output_dir = automation / "generated-tests"
    report_path = output_dir / "openhands-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    return {
        "commands": commands,
        "report": report,
        "summary": summary_path.read_text(encoding="utf-8") if summary_path.exists() else "",
        "marker": (output_dir / ".report-validated").exists(),
        "error": error,
    }


def test_main_invokes_diff_capture_with_cache_exclusions(tmp_path, monkeypatch):
    result = _run_pipeline(tmp_path, monkeypatch, _passing_report())

    assert result["error"] is None
    patch_commands = [c for c in result["commands"] if c.startswith("diff -ruN ")]
    assert patch_commands
    capture = patch_commands[-1]
    assert "--exclude=__pycache__" in capture
    assert "--exclude=.pytest_cache" in capture
    assert "/tmp/openhands-pr-baseline /workspace" in capture


def test_main_success_keeps_validation_stages_separate(tmp_path, monkeypatch):
    result = _run_pipeline(tmp_path, monkeypatch, _passing_report())

    report = result["report"]
    assert result["error"] is None
    assert report["agent_report_valid"] is True
    assert report["application_diff_valid"] is True
    assert report["report_valid"] is True
    assert report["report_validation_errors"] == []
    assert report["application_diff_errors"] == []
    # _main() removes the marker after successful completion by design.
    assert result["marker"] is False
    assert "- Agent report valid: yes" in result["summary"]
    assert "- Application diff valid: yes" in result["summary"]


def test_main_binary_diff_fails_patch_capture_with_specific_error(tmp_path, monkeypatch):
    result = _run_pipeline(tmp_path, monkeypatch, _passing_report(), BINARY_DIFF)

    error = result["error"]
    assert isinstance(error, RuntimeError)
    assert "Application diff capture failed" in str(error)
    assert "Agent report failed schema or evidence validation." not in str(error)
    report = result["report"]
    assert report["agent_report_valid"] is True
    assert report["application_diff_valid"] is False
    assert report["report_valid"] is False
    assert report["application_diff_errors"] == [gt.PATCH_CAPTURE_ERROR]
    assert report["report_validation_errors"] == []
    assert report["runner_error"] == "ValueError"
    assert result["marker"] is False
    assert "- Agent report valid: yes" in result["summary"]
    assert "- Application diff valid: no" in result["summary"]


def test_main_patch_failure_is_not_schema_failure(tmp_path, monkeypatch):
    result = _run_pipeline(tmp_path, monkeypatch, _passing_report(), BINARY_DIFF)

    report = result["report"]
    assert gt.PATCH_CAPTURE_ERROR not in report["report_validation_errors"]
    assert report["report_validation_errors"] == []
    assert gt.completion_failure_reason(report) == str(result["error"])


def test_main_fallback_preserves_original_patch_capture_error(tmp_path, monkeypatch):
    result = _run_pipeline(
        tmp_path, monkeypatch, _passing_report(), BINARY_DIFF, use_main=True
    )

    assert isinstance(result["error"], RuntimeError)
    report = result["report"]
    assert report["application_diff_errors"] == [gt.PATCH_CAPTURE_ERROR]
    assert report["report_validation_errors"] == []
    assert GENERIC_ERROR not in report["report_validation_errors"]
    assert "- Validation errors:" in result["summary"]
    assert gt.PATCH_CAPTURE_ERROR in result["summary"]
    assert GENERIC_ERROR not in result["summary"]


def test_main_rejects_invalid_agent_report(tmp_path, monkeypatch):
    invalid_report = _passing_report()
    del invalid_report["blockers"]

    result = _run_pipeline(tmp_path, monkeypatch, invalid_report)

    error = result["error"]
    assert isinstance(error, RuntimeError)
    assert str(error) == "Agent report failed schema or evidence validation."
    report = result["report"]
    assert any(
        "missing required field: blockers" in item
        for item in report["report_validation_errors"]
    )
    assert gt.PATCH_CAPTURE_ERROR not in report["report_validation_errors"]
    assert report["agent_report_valid"] is False
    assert report["application_diff_valid"] is True
    assert report["report_valid"] is False
    assert report["application_diff_errors"] == []

"""Tests for trusted generated-test validation before the integrity snapshot."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

import generate_tests as gt  # noqa: E402

COMMAND = "/workspace/.venv/bin/python -m pytest tests/generated -q"


class FakeWorkspace:
    def __init__(self, exits=None, downloads=None, outputs=None):
        self.exits = list(exits or [])
        self.outputs = list(outputs or [])
        self.commands = []
        self.downloads = downloads or {}

    def execute_command(self, command, timeout=600):
        self.commands.append(command)
        code = self.exits.pop(0) if self.exits else 1
        output = self.outputs.pop(0) if self.outputs else "output"
        return SimpleNamespace(exit_code=code, stdout=output, stderr="")

    def file_download(self, source_path, destination_path):
        content = self.downloads.get(source_path)
        if content is None:
            return SimpleNamespace(success=False, error="missing")
        Path(destination_path).write_text(json.dumps(content), encoding="utf-8")
        return SimpleNamespace(success=True, error=None)


class FakeConversation:
    def __init__(self):
        self.messages = []
        self.runs = 0

    def send_message(self, message):
        self.messages.append(message)

    def run(self):
        self.runs += 1


def test_validation_command_is_exact():
    assert gt.GENERATED_TEST_VALIDATION_COMMAND == COMMAND
    assert gt.is_valid_generated_test_validation_command(COMMAND) is True


def test_invalid_validation_command_is_rejected():
    assert gt.is_valid_generated_test_validation_command(
        "python -m pytest tests/generated -q"
    ) is False
    assert gt.is_valid_generated_test_validation_command(
        "/workspace/.venv/bin/python -m pytest -q"
    ) is False


def test_maximum_validation_attempts_is_three():
    assert gt.MAX_GENERATED_TEST_VALIDATION_ATTEMPTS == 3


def test_validation_stops_on_success():
    workspace = FakeWorkspace([1, 0])
    conversation = FakeConversation()

    result = gt.validate_generated_tests(workspace, conversation)

    assert result["valid"] is True
    assert result["attempts"] == 2
    assert workspace.commands == [COMMAND, COMMAND]
    assert conversation.runs == 1


def test_fourth_attempt_is_never_started():
    workspace = FakeWorkspace([1, 1, 1, 0])
    conversation = FakeConversation()

    result = gt.validate_generated_tests(workspace, conversation)

    assert result["valid"] is False
    assert result["attempts"] == 3
    assert len(workspace.commands) == 3
    assert conversation.runs == 2
    assert any("3 validation attempts" in error for error in result["errors"])


def test_integrity_failure_stops_correction_loop():
    workspace = FakeWorkspace([1, 1, 1])
    conversation = FakeConversation()
    calls = []

    def guard():
        calls.append(True)
        return {"unchanged": len(calls) < 2}

    result = gt.validate_generated_tests(workspace, conversation, integrity_guard=guard)

    assert result["blocked"] is True
    assert result["valid"] is False
    assert result["attempts"] == 1
    assert len(workspace.commands) == 1
    assert any("integrity guard failed" in error for error in result["errors"])


def test_fix_prompt_forbids_application_test_and_manifest_changes():
    prompt = gt.generated_test_fix_prompt(2, "FAILED tests/generated/test_x.py::test_x")

    assert "validation command" in prompt
    assert "3" in prompt  # harness enforces the count; prompt states bounds
    assert "Fix ONLY test-authoring mistakes" in prompt
    assert "test files you created" in prompt
    assert "session in tests/generated/" in prompt
    assert "must NOT modify application code" in prompt
    assert "existing repository tests" in prompt
    assert "dependency manifests" in prompt
    assert "must NOT delete, skip, rename, weaken, or remove a" in prompt
    assert "failing test" in prompt
    assert "suspected_bugs" in prompt


def test_malformed_suspected_bugs_are_rejected(tmp_path):
    workspace = FakeWorkspace(downloads={
        "/workspace/openhands-suspected-bugs.json": {
            "suspected_bugs": [
                {"failing_test": "x::y", "evidence": [], "reason": ""},
                {"failing_test": "", "evidence": ["E"], "reason": "r"},
            ]
        }
    })

    assert gt.read_suspected_bugs(workspace, tmp_path) == []


def test_genuine_suspected_bugs_are_preserved(tmp_path):
    bug = {
        "failing_test": "tests/generated/test_x.py::test_x",
        "evidence": ["E   AssertionError: expected 3, got 2"],
        "reason": "Evidence indicates a genuine application bug.",
    }
    workspace = FakeWorkspace(downloads={
        "/workspace/openhands-suspected-bugs.json": {"suspected_bugs": [bug]}
    })

    assert gt.read_suspected_bugs(workspace, tmp_path) == [bug]


def test_snapshot_is_taken_after_validation_in_source():
    source = (Path(__file__).resolve().parents[1] / ".github/scripts/generate_tests.py").read_text(
        encoding="utf-8"
    )
    validation_pos = source.index(
        "generated_test_validation = validate_generated_tests("
    )
    snapshot_pos = source.index(
        "generated_tests_manifest = capture_generated_tests_snapshot(",
        validation_pos,
    )

    assert validation_pos < snapshot_pos


def test_guard_condition_requires_validation_to_succeed_in_source():
    source = (Path(__file__).resolve().parents[1] / ".github/scripts/generate_tests.py").read_text(
        encoding="utf-8"
    )
    validation_pos = source.index(
        "generated_test_validation = validate_generated_tests("
    )
    guard_pos = source.index("and generated_tests_snapshot_error is None", validation_pos)

    assert validation_pos < guard_pos


def test_integrity_checked_before_first_validation_command():
    workspace = FakeWorkspace([0])
    calls = []

    def guard():
        calls.append(list(workspace.commands))
        return {"unchanged": True}

    result = gt.validate_generated_tests(
        workspace, FakeConversation(), integrity_guard=guard
    )

    assert result["valid"] is True
    assert calls[0] == []
    assert result["guard_checks"][0]["stage"] == "before command attempt 1"


def test_integrity_checked_after_successful_validation():
    workspace = FakeWorkspace([0])

    result = gt.validate_generated_tests(
        workspace, FakeConversation(), integrity_guard=lambda: {"unchanged": True}
    )

    stages = [item["stage"] for item in result["guard_checks"]]
    assert "after command attempt 1" in stages
    assert "before validation success" in stages
    assert result["valid"] is True


def test_integrity_checked_after_correction():
    workspace = FakeWorkspace([1, 0])

    result = gt.validate_generated_tests(
        workspace, FakeConversation(), integrity_guard=lambda: {"unchanged": True}
    )

    stages = [item["stage"] for item in result["guard_checks"]]
    assert "after command attempt 1" in stages
    assert "after correction attempt 1" in stages
    assert "before command attempt 2" in stages
    assert result["valid"] is True


def test_integrity_failure_after_success_fails_closed():
    workspace = FakeWorkspace([0])
    stages = []

    def guard():
        stages.append("call")
        return {"unchanged": len(stages) != 2}

    result = gt.validate_generated_tests(
        workspace, FakeConversation(), integrity_guard=guard
    )

    assert result["valid"] is False
    assert result["blocked"] is True
    assert any("after validation command" in error for error in result["errors"])


FAILURE_OUTPUT = (
    "FAILED tests/generated/test_x.py::test_x\n"
    "E   AssertionError: expected 3, got 2\n"
)


def _bug(**overrides):
    bug = {
        "failing_test": "tests/generated/test_x.py::test_x",
        "evidence": ["E   AssertionError: expected 3, got 2"],
        "reason": "Evidence indicates a genuine application bug.",
    }
    bug.update(overrides)
    return bug


def test_suspected_application_bug_preserves_test_and_skips_correction(tmp_path):
    workspace = FakeWorkspace(
        exits=[1],
        outputs=[FAILURE_OUTPUT],
        downloads={"/workspace/openhands-suspected-bugs.json": {
            "suspected_bugs": [_bug()]
        }},
    )
    conversation = FakeConversation()

    result = gt.validate_generated_tests(
        workspace, conversation, temp_dir=tmp_path
    )

    assert result["valid"] is False
    assert result["blocked"] is True
    assert result["classification"] == "suspected_application_bug"
    assert result["suspected_bugs"] == [_bug()]
    assert conversation.runs == 0
    assert len(workspace.commands) == 1
    assert any("suspected application bug preserved" in error for error in result["errors"])
    assert result["results"][0]["classification"] == "suspected_application_bug"


def test_suspected_bug_evidence_must_match_captured_output(tmp_path):
    workspace = FakeWorkspace(downloads={
        "/workspace/openhands-suspected-bugs.json": {
            "suspected_bugs": [_bug(evidence=["E   AssertionError: expected 99"])]
        }
    })

    assert gt.read_suspected_bugs(workspace, tmp_path, FAILURE_OUTPUT) == []


def test_suspected_bug_failing_test_must_match_captured_output(tmp_path):
    workspace = FakeWorkspace(downloads={
        "/workspace/openhands-suspected-bugs.json": {
            "suspected_bugs": [_bug(failing_test="tests/generated/test_other.py::test_other")]
        }
    })

    assert gt.read_suspected_bugs(workspace, tmp_path, FAILURE_OUTPUT) == []


def test_invalid_generated_test_receives_ordinary_correction(tmp_path):
    workspace = FakeWorkspace(
        exits=[1, 0],
        outputs=[FAILURE_OUTPUT, "1 passed"],
    )
    conversation = FakeConversation()

    result = gt.validate_generated_tests(
        workspace, conversation, temp_dir=tmp_path
    )

    assert result["valid"] is True
    assert result["classification"] == "passed"
    assert result["results"][0]["classification"] == "invalid_generated_test"
    assert conversation.runs == 1

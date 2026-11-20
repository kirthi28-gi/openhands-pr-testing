"""Focused tests for Fix 3 generated-test quarantine flow."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

import generate_tests as gt  # noqa: E402
from pr_test_harness import (  # noqa: E402
    generated_test_quarantine_id,
    run_repair_loop,
    validate_repair_diagnosis,
)

TEST_ID = "tests/generated/test_x.py::test_y"
FAILURE_OUTPUT = f"FAILED {TEST_ID}\nE AssertionError: expected 3, got 2\n"


def _report(attempt, test_id=TEST_ID, category="incorrect test"):
    return json.dumps({
        "schema_version": 1,
        "detected_stack": {"languages": ["python"]},
        "environment": {"status": "ready"},
        "test_commands": ["pytest -q"],
        "failures": [{
            "attempt": attempt,
            "category": category,
            "failing_test": test_id,
            "evidence": [test_id, "E AssertionError: expected 3, got 2"],
            "root_cause": "Generated test had an incorrect expectation.",
            "root_cause_evidence": ["E AssertionError: expected 3, got 2"],
            "repair_action": "Quarantine invalid generated test.",
        }],
    })


def _diagnosis_responses(report_text):
    return [
        SimpleNamespace(exit_code=0, stdout="", stderr=""),
        SimpleNamespace(exit_code=0, stdout=str(len(report_text)), stderr=""),
        SimpleNamespace(exit_code=0, stdout=report_text, stderr=""),
    ]


class FakeWorkspace:
    def __init__(self, results):
        self.results = iter(results)
        self.commands = []

    def execute_command(self, command, timeout=1200):
        self.commands.append(command)
        return next(self.results)


class FakeConversation:
    def __init__(self):
        self.messages = []
        self.runs = 0

    def send_message(self, message):
        self.messages.append(message)

    def run(self):
        self.runs += 1


def _failed():
    return SimpleNamespace(exit_code=1, stdout=FAILURE_OUTPUT, stderr="")


def _passed():
    return SimpleNamespace(exit_code=0, stdout="1 passed\n", stderr="")


def _suite_commands(workspace):
    return [command for command in workspace.commands if command.startswith("pytest -q")]


def test_quarantine_id_accepts_generated_test_from_output():
    diagnosis = validate_repair_diagnosis(_report(1), 1, FAILURE_OUTPUT)

    assert diagnosis["validated"] is True
    assert generated_test_quarantine_id(diagnosis, FAILURE_OUTPUT) == TEST_ID


def test_quarantine_id_rejects_non_generated_test():
    output = (
        "FAILED tests/test_math.py::test_add\n"
        "E AssertionError: expected 3, got 2\n"
    )
    report = _report(1, test_id="tests/test_math.py::test_add")
    diagnosis = validate_repair_diagnosis(report, 1, output)

    assert diagnosis["validated"] is True
    assert generated_test_quarantine_id(diagnosis, output) is None


def test_quarantine_id_rejects_fabricated_id():
    diagnosis = validate_repair_diagnosis(_report(1), 1, FAILURE_OUTPUT)

    assert diagnosis["validated"] is True
    assert generated_test_quarantine_id(diagnosis, "no pytest failures here") is None


def test_quarantine_id_rejects_unsafe_generated_path():
    unsafe = "tests/generated/../../app.py::test_x"
    output = f"FAILED {unsafe}\nE AssertionError: expected 3, got 2\n"
    report = _report(1, test_id=unsafe)
    diagnosis = validate_repair_diagnosis(report, 1, output)

    assert diagnosis["validated"] is True
    assert generated_test_quarantine_id(diagnosis, output) is None


def test_quarantine_id_rejects_non_incorrect_category():
    diagnosis = validate_repair_diagnosis(
        _report(1, category="application bug"), 1, FAILURE_OUTPUT
    )

    assert diagnosis["validated"] is True
    assert generated_test_quarantine_id(diagnosis, FAILURE_OUTPUT) is None


def test_generated_incorrect_test_is_quarantined_and_rerun():
    report = _report(1)
    workspace = FakeWorkspace([
        _failed(), *_diagnosis_responses(report),
        _passed(),
    ])
    conversation = FakeConversation()

    final, attempts = run_repair_loop(
        workspace, conversation, "pytest -q", 2
    )

    suite_commands = _suite_commands(workspace)
    assert final.exit_code == 0
    assert len(suite_commands) == 2
    assert suite_commands[0] == "pytest -q"
    assert "--deselect" in suite_commands[1]
    assert TEST_ID in suite_commands[1]
    assert attempts[0]["repair_decision"] == "test_quarantined"
    assert attempts[0]["quarantined_test"] == TEST_ID
    assert attempts[0]["quarantine_evidence"]
    assert attempts[0]["quarantine_status"] == "pending_rerun"
    assert " --deselect " not in attempts[0]["command"]
    assert conversation.runs == 1


def test_non_generated_incorrect_test_is_not_quarantined():
    output = "FAILED tests/test_math.py::test_add\nE AssertionError"
    report = _report(1, test_id="tests/test_math.py::test_add")
    workspace = FakeWorkspace([_failed(), *_diagnosis_responses(report)])
    conversation = FakeConversation()

    final, attempts = run_repair_loop(
        workspace, conversation, "pytest -q", 2
    )

    suite_commands = _suite_commands(workspace)
    assert final.exit_code == 1
    assert len(suite_commands) == 1
    assert "--deselect" not in suite_commands[0]
    assert attempts[0]["repair_decision"] == "human_review"
    assert "quarantined_test" not in attempts[0]


def test_same_test_is_not_quarantined_twice():
    report1 = _report(1)
    report2 = _report(2)
    workspace = FakeWorkspace([
        _failed(), *_diagnosis_responses(report1),
        _failed(), *_diagnosis_responses(report2),
    ])
    conversation = FakeConversation()

    final, attempts = run_repair_loop(
        workspace, conversation, "pytest -q", 2
    )

    assert final.exit_code == 1
    assert len(_suite_commands(workspace)) == 2
    assert attempts[1]["quarantine_repeated"] is True
    assert attempts[1]["repair_decision"] == "human_review"
    assert "already quarantined" in attempts[1]["repair_decision_reason"]


def test_integrity_guard_failure_blocks_quarantined_rerun():
    report = _report(1)
    workspace = FakeWorkspace([_failed(), *_diagnosis_responses(report)])
    calls = []

    def guard():
        calls.append(True)
        # Calls: before first run, after run, after diagnosis, before rerun.
        return len(calls) != 4

    final, attempts = run_repair_loop(
        workspace, FakeConversation(), "pytest -q", 2,
        generated_tests_guard=guard,
    )

    assert final.exit_code == 1
    assert len(_suite_commands(workspace)) == 1
    assert attempts[0]["repair_decision"] == "human_review"
    assert "integrity guard failed before quarantined rerun" in (
        attempts[0]["repair_decision_reason"]
    )
    assert attempts[0]["quarantine_integrity"]["unchanged"] is False


def test_quarantined_tests_are_collected_for_final_report():
    attempts = [
        {
            "attempt": 1,
            "quarantined_test": TEST_ID,
            "quarantine_reason": "Invalid generated test",
            "quarantine_evidence": [TEST_ID],
            "quarantine_status": "pending_rerun",
        },
        {"attempt": 2, "exit_code": 0},
    ]

    records = gt.quarantined_tests_from_attempts(attempts)

    assert records == [{
        "attempt": 1,
        "test": TEST_ID,
        "reason": "Invalid generated test",
        "evidence": [TEST_ID],
        "status": "pending_rerun",
    }]

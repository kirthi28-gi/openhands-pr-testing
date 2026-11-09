import json
import re
import shutil
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

from pr_test_harness import (
    ensure_failure_artifacts,
    ensure_generated_tests_unchanged,
    ensure_repository_tests_unchanged,
    generated_tests_manifest_from_archive,
    repository_test_manifest_from_archive,
    run_repair_loop,
    validate_repair_diagnosis,
)


def _agent_report(attempt, category, evidence, root_cause="Bad application behavior"):
    evidence = list(evidence)
    failing_test = "tests/test_math.py::test_add"
    for item in evidence:
        match = re.search(r"FAILED\s+(\S+::\S+)", item)
        if match:
            failing_test = match.group(1)
            break
        if "::" in item:
            failing_test = item
            break
    if failing_test not in evidence:
        evidence.insert(0, failing_test)
    root_evidence = evidence[-1]
    return json.dumps({
        "schema_version": 1,
        "detected_stack": {"languages": ["python"]},
        "environment": {"status": "ready"},
        "test_commands": ["pytest"],
        "failures": [{
            "attempt": attempt,
            "category": category,
            "failing_test": failing_test,
            "evidence": evidence,
            "root_cause": root_cause,
            "root_cause_evidence": [root_evidence],
            "repair_action": "Apply a minimal source correction",
        }],
    })


class _RepairWorkspace:
    def __init__(self, results):
        self.results = iter(results)
        self.commands = []

    def execute_command(self, command, timeout=1200):
        self.commands.append(command)
        return next(self.results)


class _Conversation:
    def __init__(self):
        self.messages = []
        self.runs = 0

    def send_message(self, message):
        self.messages.append(message)

    def run(self):
        self.runs += 1


def _result(exit_code, stdout="", stderr=""):
    return SimpleNamespace(exit_code=exit_code, stdout=stdout, stderr=stderr)


def _diagnosis_responses(report_text):
    return [
        _result(0),
        _result(0, stdout=str(len(report_text))),
        _result(0, stdout=report_text),
    ]


def test_structured_application_diagnosis_requires_matching_evidence():
    output = "FAILED tests/test_math.py::test_add\nE AssertionError: expected 3, got 2"
    good = validate_repair_diagnosis(
        _agent_report(1, "application bug", ["FAILED tests/test_math.py::test_add", "expected 3, got 2"]),
        1,
        output,
    )
    bad = validate_repair_diagnosis(
        _agent_report(1, "application bug", ["different output"]), 1, output
    )

    assert good["validated"] is True
    assert good["repair_allowed"] is True
    assert bad["validated"] is False
    assert bad["repair_allowed"] is False


def test_diagnosis_requires_actual_failing_test_and_root_cause_evidence():
    output = "FAILED tests/test_math.py::test_add\nE AssertionError: expected 3, got 2"
    wrong_test = json.loads(_agent_report(
        1, "application bug", ["tests/test_math.py::test_add", "expected 3, got 2"]
    ))
    wrong_test["failures"][0]["failing_test"] = "tests/test_math.py::test_other"
    unsupported_cause = json.loads(_agent_report(
        1, "application bug", ["tests/test_math.py::test_add", "expected 3, got 2"]
    ))
    unsupported_cause["failures"][0]["root_cause_evidence"] = ["missing source line"]

    assert not validate_repair_diagnosis(json.dumps(wrong_test), 1, output)["repair_allowed"]
    assert not validate_repair_diagnosis(json.dumps(unsupported_cause), 1, output)["repair_allowed"]


def test_application_label_conflicting_with_infrastructure_is_not_repaired():
    output = "FAILED tests/test_imports.py::test_import\nModuleNotFoundError: No module named 'vendor_lib'"
    diagnosis = validate_repair_diagnosis(
        _agent_report(1, "application bug", [
            "tests/test_imports.py::test_import",
            "ModuleNotFoundError: No module named 'vendor_lib'",
        ]),
        1,
        output,
    )

    assert diagnosis["reported_category"] == "application bug"
    assert diagnosis["category"] == "unknown"
    assert diagnosis["repair_allowed"] is False


def test_non_application_or_unknown_failures_are_sent_for_human_review():
    output = "FAILED tests/test_math.py::test_add\nE assert 2 == 3"
    for category in ("dependency", "environment", "build", "incorrect test", "unknown"):
        diagnosis = validate_repair_diagnosis(
            _agent_report(1, category, ["FAILED tests/test_math.py::test_add"]), 1, output
        )
        assert diagnosis["repair_allowed"] is False

    uncertain = validate_repair_diagnosis(
        _agent_report(1, "application bug", ["exit code 1"]), 1, "exit code 1"
    )
    assert uncertain["repair_allowed"] is False
    assert "human review" in uncertain["decision_reason"]


def test_repair_loop_repairs_only_after_valid_diagnosis_then_reruns_full_suite():
    output = "FAILED tests/test_math.py::test_add\nE AssertionError: expected 3, got 2"
    report = _agent_report(1, "application bug", ["FAILED tests/test_math.py::test_add", "expected 3, got 2"])
    workspace = _RepairWorkspace([
        _result(1, stdout=output),
        *_diagnosis_responses(report),
        _result(0, stdout="1 passed"),
    ])
    conversation = _Conversation()

    final, attempts = run_repair_loop(workspace, conversation, "full-suite", 2)

    assert final.exit_code == 0
    assert workspace.commands == [
        "full-suite",
        "test -f /workspace/openhands-report.json && test ! -L /workspace/openhands-report.json",
        "wc -c < /workspace/openhands-report.json",
        "cat -- /workspace/openhands-report.json",
        "full-suite",
    ]
    assert len(conversation.messages) == 2
    assert "Diagnose only" in conversation.messages[0]
    assert "Do not modify, delete, rename, chmod, skip, or weaken" in conversation.messages[1]
    assert attempts[0]["diagnosis"]["repair_allowed"] is True
    assert attempts[0]["repair_status"] == "repair_verified"
    assert attempts[0]["repair_decision"] == "repair_authorized"


def test_incomplete_diagnosis_gets_one_correction_turn_then_repairs():
    output = (
        "FAILED tests/generated/test_square.py::test_square_positive\n"
        "E assert 12 == 4\nE + where 12 = evaluate('square(2)')"
    )
    incomplete = json.loads(_agent_report(
        1, "application bug", [
            "FAILED tests/generated/test_square.py::test_square_positive",
            "E assert 12 == 4",
        ]
    ))
    incomplete["failures"][0].pop("root_cause_evidence")
    valid = _agent_report(1, "application bug", [
        "FAILED tests/generated/test_square.py::test_square_positive",
        "E assert 12 == 4",
        "evaluate('square(2)')",
    ])
    workspace = _RepairWorkspace([
        _result(1, stdout=output),
        *_diagnosis_responses(json.dumps(incomplete)),
        *_diagnosis_responses(valid),
        _result(0, stdout="33 passed"),
    ])
    conversation = _Conversation()

    final, attempts = run_repair_loop(workspace, conversation, "full-suite", 1)

    assert final.exit_code == 0
    assert attempts[0]["diagnosis_correction"] == "validated"
    assert attempts[0]["diagnosis"]["failing_test"] == "tests/generated/test_square.py::test_square_positive"
    assert attempts[0]["repair_status"] == "repair_verified"
    assert workspace.commands.count("full-suite") == 2
    assert len(conversation.messages) == 3  # diagnose, correct report, repair
    assert "Replace the complete `failures` entry" in conversation.messages[1]
    assert "Preserve every other entry" in conversation.messages[1]


def test_rejected_diagnosis_correction_still_blocks_application_repair():
    output = "FAILED tests/generated/test_square.py::test_square_positive\nE assert 12 == 4"
    invalid = json.loads(_agent_report(1, "application bug", [
        "FAILED tests/generated/test_square.py::test_square_positive",
        "E assert 12 == 4",
    ]))
    invalid["failures"][0].pop("root_cause_evidence")
    workspace = _RepairWorkspace([
        _result(1, stdout=output),
        *_diagnosis_responses(json.dumps(invalid)),
        *_diagnosis_responses(json.dumps(invalid)),
    ])
    conversation = _Conversation()

    final, attempts = run_repair_loop(workspace, conversation, "full-suite", 1)

    assert final.exit_code == 1
    assert attempts[0]["diagnosis_correction"] == "rejected"
    assert attempts[0]["repair_decision"] == "human_review"
    assert not any("minimal correction" in message for message in conversation.messages)


def test_diagnosis_correction_checks_both_existing_test_guards():
    output = "FAILED tests/generated/test_square.py::test_square_positive\nE assert 12 == 4"
    invalid = json.loads(_agent_report(1, "application bug", [
        "FAILED tests/generated/test_square.py::test_square_positive",
        "E assert 12 == 4",
    ]))
    invalid["failures"][0].pop("root_cause_evidence")
    valid = _agent_report(1, "application bug", [
        "FAILED tests/generated/test_square.py::test_square_positive",
        "E assert 12 == 4",
    ])

    # The production callback combines these two existing integrity guards.
    # Exercise each failure independently at the post-correction checkpoint.
    for changed_guard in ("generated", "repository"):
        calls = {"generated": 0, "repository": 0}

        def integrity_guard():
            results = {}
            for name in calls:
                calls[name] += 1
                changed = name == changed_guard and calls[name] == 4
                results[name] = {
                    "unchanged": not changed,
                    "changed_paths": [
                        "tests/generated/test_square.py"
                        if name == "generated" else "tests/test_calculator.py"
                    ] if changed else [],
                    "errors": [],
                    "restored": changed,
                }
            return {
                "unchanged": all(item["unchanged"] for item in results.values()),
                "changed_paths": [path for item in results.values() for path in item["changed_paths"]],
                "errors": [],
                "restored": any(item["restored"] for item in results.values()),
            }

        workspace = _RepairWorkspace([
            _result(1, stdout=output),
            *_diagnosis_responses(json.dumps(invalid)),
            *_diagnosis_responses(valid),
        ])
        conversation = _Conversation()
        final, attempts = run_repair_loop(
            workspace, conversation, "full-suite", 1,
            generated_tests_guard=integrity_guard,
        )

        assert final.exit_code == 1
        assert calls == {"generated": 4, "repository": 4}
        assert attempts[0]["diagnosis_correction_test_integrity"]["unchanged"] is False
        assert attempts[0]["repair_decision"] == "human_review"
        assert len(conversation.messages) == 2  # diagnosis and report correction only


def test_full_suite_is_rerun_after_each_repair_and_verifies_each_result():
    output = "FAILED tests/test_math.py::test_add\nE AssertionError: expected 3, got 2"
    workspace = _RepairWorkspace([
        _result(1, stdout=output),
        *_diagnosis_responses(_agent_report(1, "regression", ["FAILED tests/test_math.py::test_add"])),
        _result(1, stdout=output),
        *_diagnosis_responses(_agent_report(2, "regression", ["FAILED tests/test_math.py::test_add"])),
        _result(0, stdout="all tests passed"),
    ])
    conversation = _Conversation()

    final, attempts = run_repair_loop(workspace, conversation, "full-suite", 2)

    assert final.exit_code == 0
    assert workspace.commands.count("full-suite") == 3
    assert [item["repair_status"] for item in attempts[:2]] == [
        "repair_not_verified", "repair_verified",
    ]


def test_existing_test_change_during_diagnosis_blocks_repair():
    output = "FAILED tests/test_math.py::test_add\nE AssertionError: expected 3, got 2"
    report = _agent_report(1, "application bug", ["FAILED tests/test_math.py::test_add"])
    workspace = _RepairWorkspace([_result(1, stdout=output), *_diagnosis_responses(report)])
    conversation = _Conversation()
    guard_results = iter([
        {"unchanged": True},
        {"unchanged": True},
        {"unchanged": False, "changed_paths": ["tests/test_calculator.py"], "restored": True},
    ])

    final, attempts = run_repair_loop(
        workspace, conversation, "full-suite", 1,
        generated_tests_guard=lambda: next(guard_results),
    )

    assert final.exit_code == 1
    assert attempts[0]["repair_decision"] == "human_review"
    assert "tests/test_calculator.py" in attempts[0]["diagnosis_test_integrity"]["changed_paths"]
    assert len(conversation.messages) == 1


def test_repair_loop_does_not_repair_unclassified_or_infrastructure_failure():
    output = "FAILED tests/test_imports.py::test_import\nModuleNotFoundError: No module named 'vendor_lib'"
    report = _agent_report(1, "application bug", [
        "tests/test_imports.py::test_import",
        "ModuleNotFoundError: No module named 'vendor_lib'",
    ])
    workspace = _RepairWorkspace([
        _result(1, stdout=output),
        *_diagnosis_responses(report),
    ])
    conversation = _Conversation()

    final, attempts = run_repair_loop(workspace, conversation, "full-suite", 2)

    assert final.exit_code == 1
    assert len(workspace.commands) == 4
    assert workspace.commands[0] == "full-suite"
    assert workspace.commands[-1] == "cat -- /workspace/openhands-report.json"
    assert len(conversation.messages) == 1
    assert attempts[0]["repair_decision"] == "human_review"
    assert attempts[0]["diagnosis"]["category"] == "unknown"


def test_failed_repair_turn_is_reported_and_does_not_rerun_tests():
    output = "FAILED tests/test_math.py::test_add\nE AssertionError: expected 3, got 2"
    report = _agent_report(1, "application bug", ["tests/test_math.py::test_add", "expected 3, got 2"])
    workspace = _RepairWorkspace([_result(1, stdout=output), *_diagnosis_responses(report)])

    class FailedRepairConversation(_Conversation):
        def run(self):
            self.runs += 1
            if self.runs == 2:
                raise RuntimeError("repair failed")

    conversation = FailedRepairConversation()
    final, attempts = run_repair_loop(workspace, conversation, "full-suite", 2)

    assert final.exit_code == 1
    assert workspace.commands.count("full-suite") == 1
    assert attempts[0]["repair_status"] == "agent_turn_failed"
    assert attempts[0]["repair_error"] == "RuntimeError"
    assert attempts[0]["repair_decision"] == "human_review"


def test_repair_loop_does_not_repair_invalid_generated_test_classification():
    output = "FAILED tests/test_math.py::test_add\nE assert 2 == 3"
    report = _agent_report(1, "incorrect test", ["FAILED tests/test_math.py::test_add"])
    workspace = _RepairWorkspace([_result(1, stdout=output), *_diagnosis_responses(report)])
    conversation = _Conversation()

    final, attempts = run_repair_loop(workspace, conversation, "full-suite", 2)

    assert final.exit_code == 1
    assert attempts[0]["diagnosis"]["category"] == "incorrect test"
    assert attempts[0]["repair_decision"] == "human_review"
    assert len(conversation.messages) == 1


def test_repair_loop_clamps_to_three_repairs_and_diagnoses_final_failure():
    output = "FAILED tests/test_math.py::test_add\nE AssertionError: expected 3, got 2"
    responses = []
    for attempt in range(1, 5):
        responses.append(_result(1, stdout=output))
        responses.extend(_diagnosis_responses(
            _agent_report(attempt, "regression", ["FAILED tests/test_math.py::test_add"])
        ))
    workspace = _RepairWorkspace(responses)
    conversation = _Conversation()

    final, attempts = run_repair_loop(workspace, conversation, "full-suite", 20)

    assert final.exit_code == 1
    assert sum(command == "full-suite" for command in workspace.commands) == 4
    assert len(attempts) == 4
    assert attempts[-1]["repair_decision"] == "limit_reached"
    assert sum("Diagnose only" in message for message in conversation.messages) == 4
    assert sum("Make at most one minimal correction" in message for message in conversation.messages) == 3


def test_repair_loop_blocks_rerun_if_generated_tests_change_during_repair():
    output = "FAILED tests/test_math.py::test_add\nE AssertionError: expected 3, got 2"
    report = _agent_report(1, "regression", ["FAILED tests/test_math.py::test_add"])
    workspace = _RepairWorkspace([
        _result(1, stdout=output),
        *_diagnosis_responses(report),
    ])
    conversation = _Conversation()
    guard_results = iter([
        {"unchanged": True},  # before initial suite
        {"unchanged": True},  # after failed suite
        {"unchanged": True},  # after diagnosis
        {"unchanged": False, "changed_paths": ["tests/generated/test_math.py"], "restored": True},
    ])

    final, attempts = run_repair_loop(
        workspace, conversation, "full-suite", 2,
        generated_tests_guard=lambda: next(guard_results),
    )

    assert final.exit_code == 1
    assert workspace.commands[0] == "full-suite"
    assert workspace.commands[-1] == "cat -- /workspace/openhands-report.json"
    assert attempts[0]["repair_status"] == "rejected_test_files_modified"
    assert attempts[0]["repair_test_integrity"]["restored"] is True


def test_repository_test_manifest_and_guard_restore_modified_existing_test(tmp_path):
    sandbox = tmp_path / "sandbox"
    tests = sandbox / "tests"
    tests.mkdir(parents=True)
    existing = tests / "test_calculator.py"
    existing.write_text("def test_original(): assert True\n")
    generated = tests / "generated"
    generated.mkdir()
    (generated / "test_agent.py").write_text("def test_generated(): assert True\n")
    archive = tmp_path / "repository-tests.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(existing, arcname="tests/test_calculator.py")
    expected = repository_test_manifest_from_archive(archive)
    existing.write_text("def test_weakened(): assert True\n")
    workspace = _RepositorySnapshotWorkspace(sandbox, tmp_path)

    integrity = ensure_repository_tests_unchanged(
        workspace, archive, expected, tmp_path / "current.tar.gz"
    )

    assert integrity["unchanged"] is False
    assert integrity["restored"] is True
    assert existing.read_text() == "def test_original(): assert True\n"
    assert (generated / "test_agent.py").is_file()


def test_repair_loop_invalidates_a_passing_run_that_changed_generated_tests():
    workspace = _RepairWorkspace([_result(0, stdout="1 passed")])
    conversation = _Conversation()
    guard_results = iter([
        {"unchanged": True},
        {"unchanged": False, "changed_paths": ["tests/generated/test_math.py"], "restored": True},
    ])

    final, attempts = run_repair_loop(
        workspace, conversation, "full-suite", 1,
        generated_tests_guard=lambda: next(guard_results),
    )

    assert final.exit_code != 0
    assert "run result is invalid" in attempts[0]["output"]
    assert attempts[0]["exit_code"] == 1
    assert conversation.messages == []


def test_generated_test_archive_manifest_fingerprints_content_and_rejects_escape(tmp_path):
    source = tmp_path / "tests/generated"
    source.mkdir(parents=True)
    (source / "test_math.py").write_text("def test_add(): assert 1 + 1 == 2\n")
    archive_path = tmp_path / "tests.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source, arcname="tests/generated")
    manifest = generated_tests_manifest_from_archive(archive_path)
    assert "tests/generated/test_math.py" in manifest
    assert manifest["tests/generated/test_math.py"]["sha256"]

    bad_archive = tmp_path / "bad.tar.gz"
    with tarfile.open(bad_archive, "w:gz") as archive:
        member = tarfile.TarInfo("tests/generated/../../outside.py")
        member.size = 1
        import io
        archive.addfile(member, io.BytesIO(b"x"))
    try:
        generated_tests_manifest_from_archive(bad_archive)
    except ValueError as exc:
        assert "Unsafe generated-test archive path" in str(exc)
    else:
        raise AssertionError("archive traversal was accepted")


class _SnapshotWorkspace:
    def __init__(self, root, temp):
        self.root = root
        self.temp = temp

    def execute_command(self, command, timeout=120):
        if "-C /workspace tests/generated" in command:
            target = self.temp / (
                "openhands-generated-tests-current.tar.gz"
                if "current.tar.gz" in command
                else "openhands-generated-tests-snapshot.tar.gz"
            )
            with tarfile.open(target, "w:gz") as archive:
                archive.add(self.root / "tests/generated", arcname="tests/generated")
            return _result(0)
        if "tar --no-same-owner --same-permissions" in command:
            shutil.rmtree(self.root / "tests/generated", ignore_errors=True)
            with tarfile.open(self.temp / "openhands-generated-tests-snapshot.tar.gz", "r:gz") as archive:
                archive.extractall(self.root, filter="fully_trusted")
            return _result(0)
        raise AssertionError(f"Unexpected snapshot command: {command}")

    def file_download(self, source_path, destination_path):
        source = self.temp / Path(source_path).name
        shutil.copy2(source, destination_path)
        return SimpleNamespace(success=True, error=None)

    def file_upload(self, source_path, destination_path):
        shutil.copy2(source_path, self.temp / Path(destination_path).name)
        return SimpleNamespace(success=True, error=None)


def _snapshot_archive(root, archive_path):
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(root / "tests/generated", arcname="tests/generated")
    return generated_tests_manifest_from_archive(archive_path)


class _RepositorySnapshotWorkspace:
    def __init__(self, root, temp):
        self.root = root
        self.temp = temp

    def execute_command(self, command, timeout=120):
        archive_path = self.temp / "openhands-repository-tests.tar.gz"
        if "-printf '%P\\0'" in command:
            with tarfile.open(archive_path, "w:gz") as archive:
                for path in self.root.glob("tests/**/*"):
                    if path.is_file() and "generated" not in path.parts:
                        archive.add(path, arcname=path.relative_to(self.root).as_posix())
            return _result(0)
        if "-delete" in command and "tar --no-same-owner" in command:
            for path in self.root.glob("tests/**/*"):
                if path.is_file() and "generated" not in path.parts:
                    path.unlink()
            with tarfile.open(self.temp / "openhands-repository-tests-trusted.tar.gz", "r:gz") as archive:
                archive.extractall(self.root, filter="fully_trusted")
            return _result(0)
        raise AssertionError(f"Unexpected repository snapshot command: {command}")

    def file_download(self, source_path, destination_path):
        shutil.copy2(self.temp / Path(source_path).name, destination_path)
        return SimpleNamespace(success=True, error=None)

    def file_upload(self, source_path, destination_path):
        shutil.copy2(source_path, self.temp / Path(destination_path).name)
        return SimpleNamespace(success=True, error=None)


def test_generated_test_guard_restores_mutated_contents_from_trusted_snapshot(tmp_path):
    sandbox = tmp_path / "sandbox"
    tests = sandbox / "tests/generated"
    tests.mkdir(parents=True)
    test_file = tests / "test_kept.py"
    test_file.write_text("def test_original(): assert True\n")
    temp = tmp_path / "docker-tmp"
    temp.mkdir()
    snapshot = tmp_path / "snapshot.tar.gz"
    expected = _snapshot_archive(sandbox, snapshot)
    test_file.write_text("def test_weakened(): assert True\n")
    workspace = _SnapshotWorkspace(sandbox, temp)

    result = ensure_generated_tests_unchanged(
        workspace, snapshot, expected, tmp_path / "current.tar.gz"
    )

    assert result["unchanged"] is False
    assert result["restored"] is True, result
    assert result["changed_paths"] == ["tests/generated/test_kept.py"]
    assert test_file.read_text() == "def test_original(): assert True\n"


def test_initial_test_generation_may_add_tests_but_not_rewrite_existing_tests(tmp_path):
    sandbox = tmp_path / "sandbox"
    tests = sandbox / "tests/generated"
    tests.mkdir(parents=True)
    (tests / "test_existing.py").write_text("def test_original(): assert True\n")
    temp = tmp_path / "docker-tmp"
    temp.mkdir()
    snapshot = tmp_path / "snapshot.tar.gz"
    expected = _snapshot_archive(sandbox, snapshot)
    (tests / "test_new.py").write_text("def test_new(): assert True\n")
    workspace = _SnapshotWorkspace(sandbox, temp)

    result = ensure_generated_tests_unchanged(
        workspace, snapshot, expected, tmp_path / "current.tar.gz", allow_additions=True
    )

    assert result["unchanged"] is True
    assert (tests / "test_new.py").is_file()


def test_failure_artifacts_include_test_results_and_repair_history(tmp_path):
    report_path = tmp_path / "openhands-report.json"
    report_path.write_text(json.dumps({
        "result": "failed",
        "harness_result": "failed",
        "harness_final_exit_code": 1,
        "environment_ready": True,
        "validated_test_command": "pytest",
        "harness_attempts": [{"attempt": 1, "exit_code": 1}],
        "repair_history": [{"attempt": 1, "decision": "human_review"}],
    }))

    ensure_failure_artifacts(tmp_path, "RuntimeError")

    results = json.loads((tmp_path / "openhands-test-results.json").read_text())
    history = json.loads((tmp_path / "openhands-repair-history.json").read_text())
    assert results["attempts"] == [{"attempt": 1, "exit_code": 1}]
    assert history == [{"attempt": 1, "decision": "human_review"}]
    assert (tmp_path / "openhands-application.patch").is_file()


def test_sandbox_test_command_starts_with_clean_environment():
    from pr_test_harness import sandbox_test_command

    command = sandbox_test_command("pytest -q")

    assert "env -i" in command
    assert command.startswith("cd /workspace && env -i")
    assert "pytest -q" in command


def test_sandbox_test_command_does_not_forward_api_credentials():
    from pr_test_harness import sandbox_test_command

    command = sandbox_test_command("pytest")

    forbidden = [
        "OPENAI_API_KEY",
        "GITHUB_TOKEN",
        "SESSION_API_KEY",
        "LLM_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
    ]
    for secret in forbidden:
        assert secret not in command

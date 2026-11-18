"""Phase 1 regression tests: trusted Python detection, repair and recovery."""

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

import repair_engine as repair_engine_module  # noqa: E402
from environment_detector import (  # noqa: E402
    declared_test_requirement,
    declared_test_requirements,
    detect_environment,
)
from environment_manager import EnvironmentManager  # noqa: E402
from failure_classifier import classify, final_status  # noqa: E402
from repair_engine import (  # noqa: E402
    MAX_REPAIR_TIME_BUDGET_SECONDS,
    RepairEngine,
    configured_repair_limits,
)
from pr_test_harness import (  # noqa: E402
    run_environment_recovery,
    verify_test_environment,
)


class ScriptedWorkspace:
    """Simulate the sandbox with mutable Python environment state."""

    def __init__(
        self,
        venv_valid=True,
        has_pip=True,
        has_pytest=True,
        status_mode="normal",
        sandbox_files=None,
    ):
        self.state = {
            "venv_valid": venv_valid,
            "has_pip": has_pip,
            "has_pytest": has_pytest,
        }
        self.status_mode = status_mode
        self.sandbox_files = dict(sandbox_files or {})
        self.python_files = 1
        self.commands = []
        self.timeouts = []

    def execute_command(self, command, timeout=600):
        self.commands.append(command)
        self.timeouts.append(timeout)
        if "python_files" in command and "json.dumps" in command:
            return self._result(0, stdout=json.dumps({
                "files": self.sandbox_files,
                "python_files": self.python_files,
            }))
        if "OPENHANDS_PYTHON_ENV_PROBE" in command:
            return self._result(0, stdout=json.dumps({
                "python_candidates": [
                    {
                        "path": "/usr/local/bin/python3",
                        "source": "path",
                        "is_venv": False,
                        "has_pytest": False,
                        "version": [3, 13, 15],
                    },
                    {
                        "path": "/workspace/.venv/bin/python",
                        "source": "project",
                        "is_venv": True,
                        "has_pytest": self.state["has_pytest"],
                        "version": [3, 12, 16],
                    },
                ]
            }))
        if "importlib.util" in command:
            if self.status_mode == "transient":
                return self._result(1, stdout="", stderr="command failed transiently")
            if self.status_mode == "malformed":
                return self._result(0, stdout="this is not json")
            if not self.state["venv_valid"]:
                return self._result(0, stdout=json.dumps({
                    "valid": False, "reason": "pyvenv.cfg missing"
                }))
            return self._result(0, stdout=json.dumps({
                "valid": True,
                "interpreter": "/workspace/.venv/bin/python",
                "version": [3, 12, 16],
                "has_pip": self.state["has_pip"],
                "has_pytest": self.state["has_pytest"],
            }))
        if "command -v" in command:
            return self._result(0)
        if "test -f" in command:
            return self._result(1)
        if "ensurepip" in command:
            self.state["has_pip"] = True
            return self._result(0, stdout="ensurepip ok")
        if "-m venv" in command:
            self.state["venv_valid"] = True
            self.state["has_pip"] = True
            return self._result(0)
        if "rm -rf" in command:
            self.state["venv_valid"] = False
            self.state["has_pytest"] = False
            return self._result(0)
        if "pip install" in command and "pytest" in command:
            self.state["has_pytest"] = True
            return self._result(0, stdout="pytest installed")
        if "-m pip" in command:
            return self._result(0 if self.state["has_pip"] else 1, stdout="pip 24.0")
        if "import pytest" in command:
            return self._result(0 if self.state["has_pytest"] else 1)
        if "--collect-only" in command:
            if not self.state["has_pytest"]:
                return self._result(1, stderr="No module named pytest")
            return self._result(0, stdout="tests/test_example.py::test_example 1 test collected")
        if "--version" in command:
            if not self.state["has_pytest"] and "pytest" in command:
                return self._result(1, stderr="No module named pytest")
            return self._result(0, stdout="Python 3.12.16")
        return self._result(0, stdout="ok")

    @staticmethod
    def _result(code, stdout="", stderr=""):
        return SimpleNamespace(exit_code=code, stdout=stdout, stderr=stderr)


def _repo(tmp_path):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_example.py").write_text(
        "import pytest\n\ndef test_example():\n    assert True\n"
    )
    return root


def _verification(failures, interpreter="/workspace/.venv/bin/python", digest="before"):
    checks = []
    for name, output, exit_code in failures:
        checks.append({
            "name": name,
            "command": f"{interpreter} -m pytest",
            "exit_code": exit_code,
            "output": output,
            "passed": False,
        })
    return {
        "ready": False,
        "failures": [f"{item['name']}: {item['output']}" for item in checks],
        "blockers": [],
        "checks": checks,
        "python_interpreter": interpreter,
        "python_environment": {"is_venv": True, "has_pytest": False},
        "execution_plan": {"digest": digest},
        "plan_digest": digest,
    }


def test_detector_reports_python_framework_and_manifests(tmp_path):
    root = _repo(tmp_path)
    (root / "requirements.txt").write_text("pytest==8.3.3\n")
    (root / "pyproject.toml").write_text(
        '[project]\nname="demo"\nrequires-python=">=3.11"\n'
    )

    detection = detect_environment(root)

    assert detection["languages"] == ["python"]
    assert detection["python"]["detected"] is True
    assert detection["python"]["test_framework"] == "pytest"
    assert detection["python"]["package_manager"] == "pip"
    assert "requirements.txt" in detection["python"]["requirements_files"]
    assert detection["python"]["declared_python_version"] == ">=3.11"
    assert detection["services"] == []


def test_classifier_distinguishes_environment_failures():
    cases = {
        "pip_missing": "No module named pip",
        "test_framework_missing": "No module named pytest",
        "dependency_conflict": "ResolutionImpossible: conflicting dependencies",
        "system_library_missing": "libtk8.6.so: cannot open shared object file",
        "service_missing": "Connection refused",
        "application_test_failed": "FAILED tests/test_example.py::test_example",
    }
    for expected, text in cases.items():
        verification = _verification([("python_test_runner", text, 1)])
        result = classify(verification)
        assert result["category"] == expected, (expected, result)
        assert result["evidence"]


def test_pr17_bare_pytest_is_repaired_and_reverified(tmp_path, monkeypatch):
    """PR #17: bare pytest fails; project venv is valid and has pytest."""
    root = _repo(tmp_path)
    workspace = ScriptedWorkspace(venv_valid=True, has_pip=True, has_pytest=True)
    detection = detect_environment(root)
    initial = _verification(
        [("python_test_runner", "env: 'pytest': No such file or directory", 127)],
        interpreter="/usr/local/bin/python",
    )

    def verify_again():
        return verify_test_environment(workspace, root, "pytest -q")

    monkeypatch.setenv("MAX_PYTHON_ENVIRONMENT_REPAIRS", "3")
    engine = RepairEngine(
        workspace, root, detection, verify_again,
        max_attempts=3, time_budget_seconds=300,
    )
    verification, history = engine.repair(initial)

    assert verification["ready"] is True
    assert verification["recovery_status"] == "READY"
    assert verification["python_interpreter"] == "/workspace/.venv/bin/python"
    assert history[0]["plan_changed"] is True
    assert history[0]["verification_ready"] is True
    assert all(entry["result"] != "blocked" for entry in history)
    assert any("pytest -q" in item for item in workspace.commands)


def test_system_python_missing_pytest_and_venv_missing_pip_are_repaired(tmp_path):
    """Job 134720894082: system python has no pytest, venv has no pip."""
    root = _repo(tmp_path)
    workspace = ScriptedWorkspace(venv_valid=True, has_pip=False, has_pytest=False)
    detection = detect_environment(root)
    initial = _verification([
        ("python_pip_availability", "No module named pip", 1),
        ("python_test_runner", "No module named pytest", 1),
    ])

    def verify_again():
        return verify_test_environment(workspace, root, "python -m pytest -q")

    engine = RepairEngine(
        workspace, root, detection, verify_again,
        max_attempts=3, time_budget_seconds=300,
    )
    verification, history = engine.repair(initial)

    assert verification["ready"] is True
    assert history[0]["category"] == "pip_missing"
    assert any(action["action"] == "bootstrap_pip" for action in history[0]["actions"])
    assert history[-1]["verification_ready"] is True
    assert workspace.state["has_pip"] is True
    assert workspace.state["has_pytest"] is True


def test_invalid_virtualenv_is_recreated(tmp_path):
    root = _repo(tmp_path)
    workspace = ScriptedWorkspace(venv_valid=False, has_pip=False, has_pytest=False)
    detection = detect_environment(root)
    initial = _verification([
        ("python_test_runner", "virtualenv pyvenv.cfg missing; env: 'pytest': No such file or directory", 127)
    ])

    def verify_again():
        return verify_test_environment(workspace, root, "pytest -q")

    engine = RepairEngine(
        workspace, root, detection, verify_again,
        max_attempts=3, time_budget_seconds=300,
    )
    verification, history = engine.repair(initial)

    assert verification["ready"] is True
    assert any("-m venv" in command for command in workspace.commands)
    assert any("-m pip --version" in command for command in workspace.commands)
    assert workspace.state["venv_valid"] is True
    assert history[0]["category"] == "virtualenv_invalid"


def test_interpreter_mismatch_is_resolved_by_rebuilding_plan(tmp_path):
    root = _repo(tmp_path)
    workspace = ScriptedWorkspace(venv_valid=True, has_pip=True, has_pytest=True)
    detection = detect_environment(root)
    initial = _verification(
        [("python_test_runner", "No module named pytest", 1)],
        interpreter="/usr/local/bin/python",
    )

    def verify_again():
        return verify_test_environment(workspace, root, "pytest -q")

    engine = RepairEngine(
        workspace, root, detection, verify_again,
        max_attempts=3, time_budget_seconds=300,
    )
    verification, _ = engine.repair(initial)

    assert verification["ready"] is True
    assert verification["python_interpreter"] == "/workspace/.venv/bin/python"


def test_missing_dependency_installs_from_declared_manifest(tmp_path):
    root = _repo(tmp_path)
    (root / "requirements.txt").write_text("pytest==8.3.3\n")
    workspace = ScriptedWorkspace(venv_valid=True, has_pip=True, has_pytest=False)
    detection = detect_environment(root)
    initial = _verification([
        ("python_test_runner", "No module named pytest", 1)
    ])

    def verify_again():
        return verify_test_environment(workspace, root, "pytest -q")

    engine = RepairEngine(
        workspace, root, detection, verify_again,
        max_attempts=3, time_budget_seconds=300,
    )
    verification, history = engine.repair(initial)

    assert verification["ready"] is True
    assert any("-r" in command and "requirements.txt" in command for command in workspace.commands)
    assert any(action["action"] == "install_test_framework" for entry in history for action in entry["actions"])


def test_unchanged_failed_command_is_not_retried(tmp_path):
    root = _repo(tmp_path)
    workspace = ScriptedWorkspace(venv_valid=True, has_pip=True, has_pytest=True)
    detection = detect_environment(root)
    initial = _verification([
        ("python_test_runner", "env: 'pytest': No such file or directory", 127)
    ])

    def verify_again():
        return _verification([
            ("python_test_runner", "env: 'pytest': No such file or directory", 127)
        ])

    engine = RepairEngine(
        workspace, root, detection, verify_again,
        max_attempts=3, time_budget_seconds=300,
    )
    verification, history = engine.repair(initial)

    assert verification["ready"] is False
    assert verification["recovery_status"] == "BLOCKED"
    assert len(history) == 2
    assert history[-1]["result"] == "blocked"
    assert "identical failed repair" in history[-1]["reason"]


def test_repair_history_and_reports_never_expose_secrets(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    workspace = ScriptedWorkspace(venv_valid=True, has_pip=True, has_pytest=True)
    monkeypatch.setenv("REPAIR_SECRET_TOKEN", "super-secret-repair-value")
    detection = detect_environment(root)
    engine = RepairEngine(
        workspace, root, detection,
        lambda: verify_test_environment(workspace, root, "pytest -q"),
        max_attempts=1, time_budget_seconds=300,
    )
    initial = _verification([
        ("python_test_runner", "env: 'pytest': No such file or directory", 127)
    ])
    verification, history = engine.repair(initial)

    serialized = json.dumps({"verification": verification, "history": history})
    assert "super-secret-repair-value" not in serialized
    assert "REPAIR_SECRET_TOKEN" not in serialized


def test_manager_never_targets_system_python(tmp_path):
    workspace = ScriptedWorkspace(venv_valid=False, has_pip=False, has_pytest=False)
    manager = EnvironmentManager(workspace)
    result = manager.ensure_venv({})

    assert result["changed"] is True
    install_commands = [item for item in workspace.commands if "pip install" in item]
    assert install_commands or result["status"].get("valid")
    assert all("/usr/local/bin/python" not in command for command in workspace.commands)


def test_transient_inspection_failure_does_not_delete_venv():
    workspace = ScriptedWorkspace(status_mode="transient")
    manager = EnvironmentManager(workspace)

    status = manager.venv_status()
    result = manager.ensure_venv({})

    assert status["state"] == "inconclusive"
    assert result["blocked"] is True
    assert "refusing to delete" in result["reason"]
    assert not any("rm -rf" in command for command in workspace.commands)


def test_malformed_inspection_output_does_not_delete_venv():
    workspace = ScriptedWorkspace(status_mode="malformed")
    manager = EnvironmentManager(workspace)

    status = manager.venv_status()
    result = manager.ensure_venv({})

    assert status["state"] == "inconclusive"
    assert result["blocked"] is True
    assert not any("rm -rf" in command for command in workspace.commands)


def test_confirmed_invalid_venv_is_recreated():
    workspace = ScriptedWorkspace(venv_valid=False, has_pip=False, has_pytest=False)
    manager = EnvironmentManager(workspace)

    result = manager.ensure_venv({})

    assert result["changed"] is True
    state = result.get("state") or result.get("status", {}).get("state")
    assert state == "valid"
    assert any("rm -rf" in command for command in workspace.commands)
    assert any("-m venv" in command for command in workspace.commands)


def test_sandbox_manifest_changes_are_detected():
    workspace = ScriptedWorkspace(sandbox_files={
        "pyproject.toml": '[project]\nname="demo"\n',
        "tests/test_example.py": "import pytest\n",
    })
    first = detect_environment(workspace=workspace)

    assert first["evidence"]["repository_root"] == "/workspace"
    assert first["evidence"]["source"] == "sandbox"
    assert first["python"]["requirements_files"] == []
    assert first["python"]["test_framework"] == "pytest"

    workspace.sandbox_files["requirements-dev.txt"] = "pytest==9.1.0\n"
    second = detect_environment(workspace=workspace)

    assert second["python"]["requirements_files"] == ["requirements-dev.txt"]
    assert declared_test_requirement(second) == "pytest==9.1.0"


def test_stale_host_manifest_is_never_used(tmp_path):
    root = _repo(tmp_path)
    (root / "requirements.txt").write_text("pytest==8.3.3\n")
    host = detect_environment(repository_root=root)
    assert host["python"]["requirements_files"] == ["requirements.txt"]

    workspace = ScriptedWorkspace(sandbox_files={
        "tests/test_example.py": "import pytest\n",
    })
    sandbox = detect_environment(workspace=workspace)

    assert sandbox["python"]["requirements_files"] == []
    assert declared_test_requirement(sandbox) == "pytest"
    assert declared_test_requirement(host) == "pytest==8.3.3"


def test_conflicting_pytest_pins_are_not_silently_installed(tmp_path):
    root = _repo(tmp_path)
    (root / "requirements.txt").write_text("pytest==8.3.3\n")
    detection = detect_environment(repository_root=root)
    detection["python"]["pyproject_text"] = "[project.optional-dependencies]\ntest=['pytest==9.0.0']\n"

    pins = declared_test_requirements(detection)
    assert set(pins) == {"pytest==8.3.3", "pytest==9.0.0"}

    workspace = ScriptedWorkspace(has_pytest=False)
    manager = EnvironmentManager(workspace)
    result = manager.install_test_framework(detection)

    assert result["blocked"] is True
    assert "conflicting pytest pins" in result["reason"]
    assert not any("pip install" in command for command in workspace.commands)


def test_every_repair_command_receives_remaining_time_budget(tmp_path):
    workspace = ScriptedWorkspace(venv_valid=True, has_pip=False, has_pytest=False)
    root = _repo(tmp_path)
    detection = detect_environment(repository_root=root)
    initial = _verification([
        ("python_pip_availability", "No module named pip", 1)
    ])

    def verify_again():
        return {
            "ready": True,
            "failures": [],
            "checks": [],
            "execution_plan": {"digest": "after"},
            "plan_digest": "after",
        }

    engine = RepairEngine(
        workspace, root, detection, verify_again,
        max_attempts=1, time_budget_seconds=30,
    )
    engine.repair(initial)

    assert workspace.timeouts
    assert all(0 < value <= 30 for value in workspace.timeouts)


def test_overall_budget_is_clamped_and_cannot_be_bypassed(monkeypatch, tmp_path):
    monkeypatch.setenv("MAX_PYTHON_ENVIRONMENT_REPAIRS", "99")
    monkeypatch.setenv("ENVIRONMENT_REPAIR_TIME_BUDGET_SECONDS", "999999")
    attempts, budget = configured_repair_limits()

    assert attempts == 3
    assert budget == MAX_REPAIR_TIME_BUDGET_SECONDS

    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    class SlowWorkspace(ScriptedWorkspace):
        def execute_command(self, command, timeout=600):
            result = super().execute_command(command, timeout=timeout)
            if "ensurepip" in command:
                clock[0] += 35
            return result

    workspace = SlowWorkspace(venv_valid=True, has_pip=False, has_pytest=False)
    root = _repo(tmp_path)
    detection = detect_environment(repository_root=root)
    initial = _verification([
        ("python_pip_availability", "No module named pip", 1)
    ])

    engine = RepairEngine(
        workspace, root, detection,
        lambda: {"ready": False, "failures": ["not verified"], "plan_digest": "before"},
        max_attempts=3, time_budget_seconds=30,
    )
    verification, history = engine.repair(initial)

    assert verification["ready"] is False
    assert any("deadline" in str(entry.get("reason", "")) for entry in history)
    assert not any("pip install" in command for command in workspace.commands)


def test_agent_recovery_cannot_bypass_trusted_verification(monkeypatch, tmp_path):
    root = _repo(tmp_path)
    workspace = ScriptedWorkspace(venv_valid=True, has_pip=True, has_pytest=True)
    trusted_result = _verification([
        ("python_test_runner", "env: 'pytest': No such file or directory", 127)
    ], digest="trusted")

    def fake_agent_recovery(*args, **kwargs):
        return {"ready": True, "plan_digest": "agent-fake", "failures": []}, []

    monkeypatch.setattr(
        repair_engine_module, "run_environment_recovery", fake_agent_recovery
    )
    engine = RepairEngine(
        workspace, root, {}, lambda: trusted_result,
        max_attempts=1, time_budget_seconds=30,
        conversation=SimpleNamespace(send_message=lambda message: None, run=lambda: None),
    )
    initial = _verification([
        ("python_test_runner", "env: 'pytest': No such file or directory", 127)
    ])
    verification, history = engine.repair(initial)

    assert verification["ready"] is False
    assert verification["recovery_status"] == "BLOCKED"
    assert history[-1]["source"] == "agent"
    assert history[-1]["verification_ready"] is False
    assert verification["plan_digest"] == "trusted"


def test_failure_category_is_available_for_final_reports():
    verification = {
        "ready": False,
        "environment_repair_history": [
            {"category": "pip_missing", "result": "blocked"}
        ],
        "failures": ["No module named pip"],
    }
    blocked = final_status(False, 1, verification=verification)
    success = final_status(True, 0, verification=verification)
    failed = final_status(True, 1, verification=verification, report_category="application bug")

    assert blocked == {"status": "BLOCKED", "failure_category": "pip_missing"}
    assert success == {"status": "SUCCESS", "failure_category": None}
    assert failed == {"status": "FAILED", "failure_category": "application bug"}

    source = (
        Path(__file__).resolve().parents[1]
        / ".github/scripts/generate_tests.py"
    ).read_text(encoding="utf-8")
    assert '"failure_category": failure_category' in source


class TimeoutConversation:
    """Conversation whose run() times out through the supported API."""

    def __init__(self):
        self.messages = []
        self.timeouts = []
        self.interrupts = 0

    def send_message(self, message):
        self.messages.append(message)

    def run(self, timeout=None):
        self.timeouts.append(timeout)
        raise TimeoutError("agent turn timed out")

    def interrupt(self):
        self.interrupts += 1


def test_agent_turn_timeout_uses_api_and_cancels_without_skipping_integrity():
    conversation = TimeoutConversation()
    guard_calls = []

    def integrity_guard():
        guard_calls.append(True)
        return {"unchanged": True}

    verification = {"ready": False, "failures": ["boom"], "blockers": []}
    result, attempts = run_environment_recovery(
        SimpleNamespace(), conversation, verification,
        lambda: pytest.fail("verification must not run after a timed-out turn"),
        max_attempts=1,
        integrity_guard=integrity_guard,
        run_timeout=7.0,
    )

    assert result["ready"] is False
    assert attempts[0]["result"] == "timeout"
    assert attempts[0]["turn"]["timeout_enforcement"] == "api_timeout"
    assert attempts[0]["turn"]["timeout_seconds"] == 7.0
    assert conversation.timeouts == [7.0]
    assert conversation.interrupts == 1
    assert len(guard_calls) == 2  # before and after the cancelled turn


def test_agent_turn_timeout_cannot_skip_a_failed_integrity_check():
    conversation = TimeoutConversation()
    guard_results = [{"unchanged": True}, {"unchanged": False, "changed_paths": ["tests/generated/x.py"]}]

    def integrity_guard():
        return guard_results.pop(0)

    verification = {"ready": False, "failures": ["boom"], "blockers": []}
    result, attempts = run_environment_recovery(
        SimpleNamespace(), conversation, verification,
        lambda: pytest.fail("verification must not run when integrity fails"),
        max_attempts=1,
        integrity_guard=integrity_guard,
        run_timeout=5.0,
    )

    assert attempts[0]["result"] == "blocked"
    assert "integrity" in attempts[0]["reason"]
    assert conversation.interrupts == 1


def test_agent_recovery_without_api_timeout_is_recorded_not_crashed():
    conversation = SimpleNamespace(send_message=lambda message: None, run=lambda: None)
    verification = {"ready": False, "failures": ["boom"], "blockers": []}
    result, attempts = run_environment_recovery(
        SimpleNamespace(), conversation, verification,
        lambda: {"ready": True, "failures": []},
        max_attempts=1,
        run_timeout=5.0,
    )

    assert result["ready"] is True
    assert attempts[0]["turn"]["timeout_enforcement"] == "unsupported_by_conversation"
    assert attempts[0]["turn"]["timeout_supported"] is False


def test_repair_engine_passes_remaining_budget_and_never_starts_a_second_turn(
    monkeypatch, tmp_path,
):
    root = _repo(tmp_path)
    workspace = ScriptedWorkspace(venv_valid=True, has_pip=True, has_pytest=True)
    captured = []
    trusted_results = []

    def fake_agent_recovery(*args, **kwargs):
        captured.append(kwargs)
        return {
            "ready": False,
            "failures": ["Connection refused"],
            "plan_digest": "unchanged",
        }, []

    monkeypatch.setattr(
        repair_engine_module, "run_environment_recovery", fake_agent_recovery
    )
    engine = RepairEngine(
        workspace, root, {},
        lambda: {"ready": False, "failures": ["Connection refused"], "plan_digest": "unchanged"},
        max_attempts=3, time_budget_seconds=30,
        conversation=SimpleNamespace(send_message=lambda m: None, run=lambda: None),
    )
    initial = {
        "ready": False,
        "failures": ["Connection refused"],
        "blockers": [],
        "checks": [],
        "plan_digest": "unchanged",
    }
    verification, history = engine.repair(initial)

    assert verification["ready"] is False
    assert len(captured) == 1
    assert 0 < captured[0]["run_timeout"] <= 30
    assert captured[0]["max_attempts"] == 1
    assert captured[0]["run_timeout"] < 30  # reserve retained for verification
    assert sum(1 for item in history if item.get("source") == "agent") == 1

    second, second_entry = engine._agent_recovery(verification)
    assert second_entry["result"] == "blocked"
    assert "already used" in second_entry["reason"]

"""Regression tests for the single test-execution-plan architecture.

These cover the PR #17 failure mode: an agent prepared ``/workspace/.venv`` and
wrote a bare ``pytest`` command, but the harness executed verification under a
sanitized environment where bare ``pytest`` resolved to nothing (exit 127).
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

from pr_test_harness import (
    EnvironmentResolutionError,
    TestExecutionPlan,
    resolve_test_execution_plan,
    run_environment_recovery,
    run_repair_loop,
    sandbox_test_command,
    test_execution_plan_from_command as build_plan_from_command,
    verify_test_environment,
)

# Reuse the existing local probe doubles so the real discovery script is
# exercised, including arbitrary and nested virtual-environment names.
from test_pr_test_harness import (
    _LocalProbeWorkspace,
    _clean_probe_env,
    _write_stub_interpreter,
)


def _result(code, stdout="", stderr=""):
    return SimpleNamespace(exit_code=code, stdout=stdout, stderr=stderr)


class _PlanWorkspace:
    """Return a fixed interpreter-probe report and simulate check outcomes."""

    def __init__(self, report=None, fail_markers=(), probe_failure=False):
        self.report = report
        self.fail_markers = tuple(fail_markers)
        self.probe_failure = probe_failure
        self.commands = []

    def execute_command(self, command, timeout=600):
        self.commands.append(command)
        if "OPENHANDS_PYTHON_ENV_PROBE" in command:
            if self.probe_failure or self.report is None:
                return _result(1, stderr="probe unavailable")
            return _result(0, stdout=json.dumps(self.report))
        if any(marker in command for marker in self.fail_markers):
            return _result(
                1,
                stderr="/workspace/.venv/bin/python: No module named pytest",
            )
        return _result(0, stdout="Python 3.12.16\n1 test collected\n")


class _Conversation:
    def __init__(self):
        self.messages = []
        self.runs = 0

    def send_message(self, message):
        self.messages.append(message)

    def run(self):
        self.runs += 1


def _python_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_example.py").write_text(
        "def test_example():\n    assert True\n"
    )
    return root


_PR17_REPORT = {
    "python_candidates": [
        {
            "path": "/usr/local/bin/python3",
            "source": "path",
            "has_pytest": False,
            "is_venv": False,
            "version": [3, 13, 15],
        },
        {
            "path": "/workspace/.venv/bin/python",
            "source": "project",
            "has_pytest": True,
            "is_venv": True,
            "version": [3, 12, 16],
        },
    ]
}


def test_pr17_bare_pytest_uses_project_venv_for_every_check(tmp_path, monkeypatch):
    """Exact PR #17 scenario: bare pytest plus a prepared project venv."""
    root = _python_repo(tmp_path)
    workspace = _PlanWorkspace(_PR17_REPORT)
    # Host environment contamination must never select the interpreter.
    monkeypatch.setenv("VIRTUAL_ENV", "/host/leaked-venv")
    monkeypatch.setenv("PYTHONPATH", "/host/leaked-path")
    interpreter = "/workspace/.venv/bin/python"

    verification = verify_test_environment(workspace, root, "pytest -q")

    assert verification["ready"] is True
    assert verification["python_interpreter"] == interpreter
    assert verification["python_environment"]["source"] == "project"
    assert verification["execution_plan"]["interpreter"] == interpreter
    assert verification["execution_plan"]["argv"] == [
        interpreter, "-m", "pytest", "-q"
    ]

    # No verification step may invoke pytest by name under the sanitized PATH.
    for check in verification["checks"]:
        if check["command"] is None:
            continue
        assert "env -i" in check["command"], check["name"]
        if "pytest" in check["command"]:
            assert "-m pytest" in check["command"], check["name"]
            assert interpreter in check["command"], check["name"]
    for command in workspace.commands:
        if "pytest" in command and "OPENHANDS_PYTHON_ENV_PROBE" not in command:
            assert "-m pytest" in command, command

    plan = TestExecutionPlan.from_dict(verification["execution_plan"])
    suite_command = sandbox_test_command(plan)
    assert f"{interpreter} -m pytest -q" in suite_command
    assert 'VIRTUAL_ENV="/workspace/.venv"' in suite_command
    assert 'PATH="/workspace/.venv/bin:${PATH:-' in suite_command
    assert "/usr/local/bin/python3" not in suite_command
    assert "/host/leaked-venv" not in suite_command
    assert "/host/leaked-path" not in suite_command


def test_verification_and_final_execution_share_one_plan(tmp_path):
    """The same plan digest must drive checks and the executed suite."""
    root = _python_repo(tmp_path)
    workspace = _PlanWorkspace(_PR17_REPORT)
    interpreter = "/workspace/.venv/bin/python"

    verification = verify_test_environment(workspace, root, "pytest -q")
    plan = TestExecutionPlan.from_dict(verification["execution_plan"])
    suite_command = sandbox_test_command(plan)

    assert verification["plan_digest"] == plan.digest
    for check in verification["checks"]:
        if check["command"] is None:
            continue
        assert interpreter in check["command"], check["name"]

    conversation = _Conversation()
    final, attempts = run_repair_loop(workspace, conversation, plan, 0)

    assert final.exit_code == 0
    assert attempts[0]["command"] == suite_command
    assert attempts[0]["plan_digest"] == plan.digest
    assert attempts[0]["execution_plan"]["argv"] == list(plan.argv)


def test_project_venv_without_pytest_never_borrows_system_pytest(tmp_path):
    """A prepared project environment wins; missing pytest is a blocker."""
    root = _python_repo(tmp_path)
    interpreter = "/workspace/.venv/bin/python"
    workspace = _PlanWorkspace(
        {
            "python_candidates": [
                {
                    "path": interpreter,
                    "source": "project",
                    "has_pytest": False,
                    "is_venv": True,
                },
                {
                    "path": "/usr/local/bin/python3",
                    "source": "system",
                    "has_pytest": True,
                    "is_venv": False,
                },
            ]
        },
        fail_markers=(interpreter,),
    )

    verification = verify_test_environment(workspace, root, "pytest -q")

    assert verification["ready"] is False
    assert verification["python_interpreter"] == interpreter
    assert verification["python_environment"]["has_pytest"] is False
    assert verification["execution_plan"]["interpreter"] == interpreter
    assert any(
        "refusing to borrow" in failure for failure in verification["failures"]
    )
    plan = TestExecutionPlan.from_dict(verification["execution_plan"])
    suite_command = sandbox_test_command(plan)
    assert f"{interpreter} -m pytest" in suite_command
    assert "/usr/local/bin/python3 -m pytest" not in suite_command


def test_absolute_workspace_pytest_path_plans_sibling_interpreter():
    command = sandbox_test_command("/workspace/custom/bin/pytest -q")

    assert "/workspace/custom/bin/python -m pytest -q" in command
    assert 'VIRTUAL_ENV="/workspace/custom"' in command
    assert 'PATH="/workspace/custom/bin:${PATH:-' in command


def test_absolute_workspace_interpreter_is_resolved_as_a_plan(tmp_path):
    root = _python_repo(tmp_path)
    interpreter = "/workspace/custom/bin/python"
    workspace = _PlanWorkspace(
        {
            "python_candidates": [
                {
                    "path": interpreter,
                    "source": "project",
                    "has_pytest": True,
                    "is_venv": True,
                },
            ]
        }
    )

    plan = resolve_test_execution_plan(
        workspace, root, "/workspace/custom/bin/python -m pytest -q"
    )

    assert plan.interpreter == interpreter
    assert plan.venv_root == "/workspace/custom"
    assert plan.argv == (interpreter, "-m", "pytest", "-q")
    assert plan.to_dict()["interpreter"].startswith("/workspace/")


def test_nested_arbitrarily_named_venv_is_discovered(tmp_path):
    """Discovery must not depend on ``.venv``/``venv`` naming or depth."""
    root = tmp_path / "project"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_example.py").write_text(
        "def test_example():\n    assert True\n"
    )
    environment = root / "tools" / "qa" / "ci-venv"
    stub = environment / "bin" / "python"
    _write_stub_interpreter(stub)
    (environment / "pyvenv.cfg").write_text("home = /usr/bin\n")

    workspace = _LocalProbeWorkspace(str(root), env=_clean_probe_env())
    plan = resolve_test_execution_plan(workspace, root, "pytest -q")

    assert plan.interpreter == str(stub)
    assert plan.venv_root == str(environment)
    assert plan.source == "project"
    assert plan.evidence["python_environment"]["candidates"]


@pytest.mark.parametrize(
    "command",
    [
        "uv run --offline --no-sync -- pytest -q",
        "poetry run pytest -q",
        "pipenv run pytest -q",
    ],
)
def test_wrapped_runners_keep_their_own_environment(tmp_path, command):
    root = _python_repo(tmp_path)
    workspace = _PlanWorkspace({"python_candidates": []})

    plan = resolve_test_execution_plan(workspace, root, command)
    verification = verify_test_environment(workspace, root, command, plan=plan)

    assert plan.source == "wrapper"
    assert plan.interpreter is None
    assert plan.venv_root is None
    assert plan.argv == tuple(command.split())
    assert verification["ready"] is True
    assert verification["plan_digest"] == plan.digest
    assert sandbox_test_command(plan).endswith(" " + command)
    for check in verification["checks"]:
        if check["command"] is None:
            continue
        assert "/workspace/.venv/bin/python" not in check["command"]
    # Wrappers were never bypassed by the interpreter probe.
    assert all("OPENHANDS_PYTHON_ENV_PROBE" not in item for item in workspace.commands)


def test_recovery_records_a_changed_execution_plan():
    conversation = _Conversation()
    initial = {
        "ready": False,
        "failures": ["python_test_runner: missing"],
        "blockers": ["python_test_runner: missing"],
        "plan_digest": "plan-a",
        "plan_source": "system",
        "execution_plan": {"digest": "plan-a"},
    }
    recovered = {
        "ready": True,
        "failures": [],
        "blockers": [],
        "plan_digest": "plan-b",
        "plan_source": "project",
        "execution_plan": {"digest": "plan-b"},
    }

    verification, attempts = run_environment_recovery(
        object(), conversation, initial, lambda: recovered, max_attempts=1,
        sleep_fn=lambda _seconds: None,
    )

    assert verification["ready"] is True
    assert attempts[0]["plan_changed"] is True
    assert attempts[0]["previous_plan_digest"] == "plan-a"
    assert attempts[0]["plan_digest"] == "plan-b"
    assert verification["plan_changes"] == [{
        "attempt": 1,
        "previous_plan_digest": "plan-a",
        "plan_digest": "plan-b",
        "plan_source": "project",
    }]


def test_recovery_records_an_unchanged_execution_plan():
    conversation = _Conversation()
    initial = {
        "ready": False,
        "failures": ["still failing"],
        "blockers": ["still failing"],
        "plan_digest": "plan-a",
        "plan_source": "project",
    }
    recovered = {
        "ready": False,
        "failures": ["still failing"],
        "blockers": ["still failing"],
        "plan_digest": "plan-a",
        "plan_source": "project",
    }

    verification, attempts = run_environment_recovery(
        object(), conversation, initial, lambda: recovered, max_attempts=1,
        sleep_fn=lambda _seconds: None,
    )

    assert attempts[0]["plan_changed"] is False
    assert verification["plan_changes"] == []


def test_unresolved_environment_fails_safely(tmp_path):
    root = _python_repo(tmp_path)
    workspace = _PlanWorkspace({"python_candidates": []})

    verification = verify_test_environment(workspace, root, "pytest -q")

    assert verification["ready"] is False
    assert verification["execution_plan"] is None
    assert verification["plan_digest"] is None
    assert verification["checks"] == []
    assert any(
        "environment_unresolved" in failure for failure in verification["failures"]
    )
    assert verification["environment_unresolved"]["tokens"] == ["pytest", "-q"]
    # Only the probe ran; no bare pytest command was executed.
    assert all("OPENHANDS_PYTHON_ENV_PROBE" in item for item in workspace.commands)


def test_unresolved_environment_cannot_be_rendered_as_a_command():
    with pytest.raises(EnvironmentResolutionError):
        sandbox_test_command("pytest -q")


def test_raw_python_command_is_rejected_by_the_repair_loop():
    with pytest.raises(EnvironmentResolutionError):
        run_repair_loop(object(), _Conversation(), "pytest -q", 0)


def test_tampered_execution_plan_digest_is_rejected():
    plan = build_plan_from_command(
        "/workspace/.venv/bin/python -m pytest -q"
    )
    data = plan.to_dict()
    data["argv"] = ["/usr/local/bin/python", "-m", "pytest", "-q"]

    with pytest.raises(ValueError, match="digest does not match"):
        TestExecutionPlan.from_dict(data)


def test_plan_without_an_interpreter_fails_safely(tmp_path):
    root = _python_repo(tmp_path)
    workspace = _PlanWorkspace(_PR17_REPORT)
    incomplete = TestExecutionPlan(
        argv=("/workspace/.venv/bin/python", "-m", "pytest", "-q"),
        env={"PATH": "/usr/local/bin"},
        interpreter=None,
        venv_root=None,
        framework="pytest",
        ecosystem="python",
        source="command",
    )

    verification = verify_test_environment(workspace, root, "pytest -q", plan=incomplete)

    assert verification["ready"] is False
    assert verification["execution_plan"] is None
    assert verification["checks"] == []
    assert "environment_unresolved" in verification["failures"][0]


def test_probe_failure_is_reported_as_environment_unresolved(tmp_path):
    root = _python_repo(tmp_path)
    workspace = _PlanWorkspace(probe_failure=True)

    verification = verify_test_environment(workspace, root, "pytest -q")

    assert verification["ready"] is False
    assert "environment_unresolved" in verification["failures"][0]
    assert all("OPENHANDS_PYTHON_ENV_PROBE" in item for item in workspace.commands)


def test_execution_plan_serialization_is_secret_free(tmp_path):
    root = _python_repo(tmp_path)
    workspace = _PlanWorkspace(_PR17_REPORT)
    os.environ["TESTSECRET_TOKEN"] = "super-secret-value"
    try:
        verification = verify_test_environment(workspace, root, "pytest -q")
        serialized = json.dumps(verification["execution_plan"])
    finally:
        os.environ.pop("TESTSECRET_TOKEN", None)

    assert "super-secret-value" not in serialized
    assert "TESTSECRET_TOKEN" not in serialized
    assert verification["execution_plan"]["digest"] == verification["plan_digest"]

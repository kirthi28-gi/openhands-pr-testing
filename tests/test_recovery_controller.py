import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

import recovery_controller
from pr_test_harness import run_environment_recovery, _redact_sensitive_values


class FakeConversation:
    def __init__(self):
        self.messages = []
        self.runs = 0

    def send_message(self, message):
        self.messages.append(message)

    def run(self):
        self.runs += 1


@pytest.mark.parametrize("failure", [
    "NameError: name 'SimpleNamespace' is not defined",
    "ModuleNotFoundError: No module named 'known_dependency'",
    "ImportError: local package path unresolved",
    "npm ERR! missing package from package-lock.json",
    "SyntaxError: invalid syntax in application module",
    "BUILD FAILED: Java compilation error",
])
def test_environment_recovery_retries_after_evidence_led_fix(failure):
    conversation = FakeConversation()
    calls = []

    def verify_again():
        calls.append(True)
        return {
            "ready": True,
            "failures": [],
            "blockers": [],
            "ecosystem": "detected-from-fixture",
        }

    initial = {"ready": False, "failures": [failure], "blockers": [failure]}
    result, attempts = run_environment_recovery(
        SimpleNamespace(), conversation, initial, verify_again,
        max_attempts=2, sleep_fn=lambda _: None,
    )

    assert result["ready"] is True
    assert attempts[0]["result"] == "ready"
    assert len(calls) == conversation.runs == 1
    assert failure in conversation.messages[0]
    assert "Do not modify, delete, skip" in conversation.messages[0]
    assert "pip/uv" in conversation.messages[0]
    assert "npm/pnpm/yarn" in conversation.messages[0]
    assert "Maven/Gradle" in conversation.messages[0]
    assert "Preserve pinned versions and lockfiles" in conversation.messages[0]
    assert "use the declared manager" in conversation.messages[0]
    assert "Do not install packages" not in conversation.messages[0]


def test_environment_recovery_is_bounded_when_installation_keeps_failing():
    conversation = FakeConversation()
    verification = {
        "ready": False,
        "failures": ["pip install evidenced-package exited 1: no matching distribution"],
        "blockers": [],
    }
    calls = []

    def verify_again():
        calls.append(True)
        return {
            "ready": False,
            "failures": ["pip install evidenced-package exited 1: no matching distribution"],
            "blockers": ["installer output retained"],
        }

    result, attempts = run_environment_recovery(
        SimpleNamespace(), conversation, verification, verify_again,
        max_attempts=2, sleep_fn=lambda _: None,
    )

    assert result["ready"] is False
    assert len(attempts) == len(calls) == conversation.runs == 2
    assert [item["attempt"] for item in attempts] == [1, 2]


@pytest.mark.parametrize("failure", [
    "Docker is not available: permission denied",
    "Authentication failed: credentials unavailable",
    "Connection refused by required external service",
])
def test_nonrecoverable_environment_boundary_stops_without_agent_edit(failure):
    conversation = FakeConversation()
    verification = {"ready": False, "failures": [failure], "blockers": [failure]}
    result, attempts = run_environment_recovery(
        SimpleNamespace(), conversation, verification, lambda: pytest.fail("must not retry"),
        max_attempts=2,
    )

    assert result["ready"] is False
    assert len(attempts) == 1 and attempts[0]["result"] == "blocked"
    assert conversation.runs == 0


def test_recovery_policies_are_separate_and_repo_recovery_allows_manager_installs():
    secret = "orchestration-secret-value"
    prompt = recovery_controller._orchestration_recovery_prompt(
        f"failure output contains {secret}", {"CUSTOM_API_KEY": secret}
    )
    harness_path = Path(recovery_controller.__file__).with_name("pr_test_harness.py")
    environment_prompt_source = harness_path.read_text(encoding="utf-8")

    assert "Do not install packages" in prompt
    assert secret not in prompt
    assert "pip/uv for Python" in environment_prompt_source
    assert "npm/pnpm/yarn for JavaScript" in environment_prompt_source
    assert "Maven/Gradle for" in environment_prompt_source
    assert "Do not install packages" not in environment_prompt_source


def test_sensitive_environment_values_are_redacted_from_controller_and_harness(monkeypatch):
    secret = "test-api-token-super-secret"
    controller_text = recovery_controller._redact_secrets(
        f"request failed using {secret}", {"CUSTOM_REGISTRY_TOKEN": secret}
    )
    monkeypatch.setenv("CUSTOM_REGISTRY_TOKEN", secret)
    harness_text = _redact_sensitive_values(f"installer printed {secret}")

    assert secret not in controller_text
    assert secret not in harness_text
    assert "[REDACTED]" in controller_text and "[REDACTED]" in harness_text


def test_controller_resolves_trusted_root_from_its_file_not_workspace_parent(monkeypatch, tmp_path):
    trusted = tmp_path / "trusted"
    scripts = trusted / ".github/scripts"
    scripts.mkdir(parents=True)
    controller_file = scripts / "recovery_controller.py"
    controller_file.write_text("# test location")
    monkeypatch.setattr(recovery_controller, "__file__", str(controller_file))
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    roots = []
    monkeypatch.setattr(recovery_controller, "run_controller", lambda root: roots.append(root) or 0)

    with pytest.raises(SystemExit) as exit_info:
        recovery_controller.main()

    assert exit_info.value.code == 0
    assert roots == [trusted]


def test_trusted_generator_keeps_workflow_inputs_but_filters_host_credentials():
    workflow_env = {
        "GITHUB_WORKSPACE": "/runner/work/repo/repo",
        "AUTOMATION_DIR": "/runner/work/repo/repo/trusted",
        "PR_SOURCE_DIR": "/runner/work/repo/repo/pr-source",
        "PR_DIFF_FILE": "/runner/work/repo/repo/trusted/pr.diff",
        "PR_TITLE": "A change",
        "PR_BODY": "Details",
        "PR_NUMBER": "12",
        "PR_BASE_SHA": "base",
        "PR_HEAD_SHA": "head",
        "GITHUB_STEP_SUMMARY": "/runner/summary",
        "LLM_API_KEY": "host-only-key",
        "LLM_MODEL": "model-name",
        "LLM_BASE_URL": "https://example.invalid/v1",
        "GITHUB_TOKEN": "must-be-removed",
    }
    child_env = recovery_controller._trusted_runner_environment(workflow_env)

    assert {key: child_env[key] for key in workflow_env if key != "GITHUB_TOKEN"} == {
        key: value for key, value in workflow_env.items() if key != "GITHUB_TOKEN"
    }
    assert "GITHUB_TOKEN" not in child_env


def test_environment_recovery_blocks_when_agent_changes_protected_tests():
    conversation = FakeConversation()
    checks = iter([
        {"unchanged": True},
        {"unchanged": False, "changed_paths": ["tests/test_original.py"]},
    ])
    result, attempts = run_environment_recovery(
        SimpleNamespace(), conversation,
        {"ready": False, "failures": ["missing dependency"], "blockers": []},
        lambda: pytest.fail("verification must not run after test tampering"),
        max_attempts=1, integrity_guard=lambda: next(checks),
    )

    assert result["ready"] is False
    assert attempts[0]["result"] == "blocked"
    assert attempts[0]["integrity"]["changed_paths"] == ["tests/test_original.py"]


class FakeDockerWorkspace:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.commands = []
        self.archive_members = []
        self.cleaned = False

    def file_upload(self, source_path, destination_path):
        with tarfile.open(source_path, "r:gz") as archive:
            self.archive_members = archive.getnames()
        return SimpleNamespace(success=True, error=None)

    def execute_command(self, command):
        self.commands.append(command)
        if "py_compile" in command:
            return SimpleNamespace(exit_code=0, stdout="recovered orchestration imports passed", stderr="")
        return SimpleNamespace(exit_code=0, stdout="extracted", stderr="")

    def cleanup(self):
        self.cleaned = True


def test_orchestration_controller_retries_staged_import_only_in_fresh_docker(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    scripts = trusted / ".github/scripts"
    scripts.mkdir(parents=True)
    for name in ("generate_tests.py", "pr_test_harness.py", "safe_extract.py"):
        (scripts / name).write_text("# trusted baseline\n")
    automation = tmp_path / "artifacts"
    env = {
        "AUTOMATION_DIR": str(automation),
        "GITHUB_WORKSPACE": str(tmp_path),
        "LLM_API_KEY": "host-only-llm-key",
        "GITHUB_TOKEN": "github-token-must-not-forward",
        "GH_TOKEN": "gh-token-must-not-forward",
        "ACTIONS_RUNTIME_TOKEN": "runtime-token-must-not-forward",
        "AWS_SECRET_ACCESS_KEY": "host-cloud-secret-must-not-forward",
    }
    runs = []

    def run(script, _env, timeout):
        assert timeout == 30 * 60
        assert _env["LLM_API_KEY"] == "host-only-llm-key"
        assert "GITHUB_TOKEN" not in _env
        assert "GH_TOKEN" not in _env
        assert "ACTIONS_RUNTIME_TOKEN" not in _env
        assert "AWS_SECRET_ACCESS_KEY" not in _env
        runs.append(script)
        return SimpleNamespace(returncode=1, stdout="", stderr="Traceback: orchestration crash")

    def recover(_root, _failure, staged_script, _env):
        staged_script.write_text("# validated staged recovery\n")

    monkeypatch.setattr(recovery_controller, "_run_generator", run)
    monkeypatch.setattr(recovery_controller, "_recover_staged_script", recover)
    monkeypatch.setattr(recovery_controller, "_normal_result_exists", lambda _: False)

    workspaces = []

    def workspace_factory(**kwargs):
        workspace = FakeDockerWorkspace(**kwargs)
        workspaces.append(workspace)
        return workspace

    assert recovery_controller.run_controller(trusted, env, workspace_factory) == 1
    assert len(runs) == 1
    assert runs[0] == scripts / "generate_tests.py"
    assert (scripts / "generate_tests.py").read_text() == "# trusted baseline\n"
    report = (automation / "generated-tests/orchestration-recovery.json").read_text()
    assert '"result": "sandbox_stage_passed"' in report
    assert '"pipeline_resumed": false' in report
    assert len(workspaces) == 1 and workspaces[0].cleaned
    assert workspaces[0].kwargs["network"] == "bridge"
    assert workspaces[0].kwargs["volumes"] == []
    assert "LLM_API_KEY" not in workspaces[0].kwargs["forward_env"]
    assert "GITHUB_TOKEN" not in workspaces[0].kwargs["forward_env"]
    assert "privileged" not in workspaces[0].kwargs
    assert workspaces[0].archive_members == [
        "scripts/generate_tests.py", "scripts/pr_test_harness.py"
    ]
    probe = workspaces[0].commands[-1]
    assert "env -u SESSION_API_KEY" in probe
    assert "-u LLM_API_KEY -u GITHUB_TOKEN -u GH_TOKEN" in probe
    assert 'assert hasattr(generate_tests, "SimpleNamespace")' in probe


def test_recovered_runner_accepts_missing_standard_library_import_only():
    baseline = "import os\n\ndef run():\n    return os.getcwd()\n"
    fixed = "import os\nfrom types import SimpleNamespace\n\ndef run():\n    return os.getcwd()\n"

    recovery_controller._validate_recovered_candidate(baseline, fixed)


def test_recovered_runner_accepts_only_the_exact_trusted_script_path_setup():
    baseline = "import sys\nfrom pathlib import Path\n\ndef run():\n    return True\n"
    fixed = (
        "import sys\nfrom pathlib import Path\n\n"
        "SCRIPT_DIR = Path(__file__).resolve().parent\n"
        "if str(SCRIPT_DIR) not in sys.path:\n"
        "    sys.path.insert(0, str(SCRIPT_DIR))\n\n"
        "def run():\n    return True\n"
    )

    recovery_controller._validate_recovered_candidate(baseline, fixed)


def test_recovered_runner_rejects_logic_changes_that_could_weaken_safeguards():
    baseline = "import os\n\ndef run():\n    return os.getcwd()\n"
    unsafe = "import os\n\ndef run():\n    return os.getenv('LLM_API_KEY')\n"

    with pytest.raises(RuntimeError, match="executable logic"):
        recovery_controller._validate_recovered_candidate(baseline, unsafe)


def test_recovered_runner_rejects_unapproved_imports():
    baseline = "import os\n\ndef run():\n    return os.getcwd()\n"
    unsafe = "import os\nimport urllib\n\ndef run():\n    return os.getcwd()\n"

    with pytest.raises(RuntimeError, match="unapproved imports"):
        recovery_controller._validate_recovered_candidate(baseline, unsafe)


def test_orchestration_controller_does_not_bypass_validated_test_failure(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    (trusted / ".github/scripts").mkdir(parents=True)
    automation = tmp_path / "artifacts"
    calls = []
    monkeypatch.setattr(
        recovery_controller, "_run_generator",
        lambda *_, **__: (calls.append(True) or SimpleNamespace(returncode=1, stdout="test failed", stderr="")),
    )
    monkeypatch.setattr(recovery_controller, "_normal_result_exists", lambda _: True)
    monkeypatch.setattr(recovery_controller, "_recover_staged_script", lambda *args: pytest.fail("must not recover"))

    assert recovery_controller.run_controller(
        trusted, {"AUTOMATION_DIR": str(automation), "GITHUB_WORKSPACE": str(tmp_path)}
    ) == 1
    assert len(calls) == 1


def test_controller_reports_docker_startup_boundary_without_recursive_recovery(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    (trusted / ".github/scripts").mkdir(parents=True)
    output = tmp_path / "artifacts/generated-tests"
    monkeypatch.setattr(
        recovery_controller, "_run_generator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("Docker is not available: permission denied to daemon socket")
        ),
    )
    monkeypatch.setattr(
        recovery_controller, "_recover_staged_script",
        lambda *args: pytest.fail("controller must not recursively recover its own Docker boundary"),
    )

    assert recovery_controller.run_controller(
        trusted,
        {"AUTOMATION_DIR": str(tmp_path / "artifacts"), "GITHUB_WORKSPACE": str(tmp_path)},
    ) == 1
    report = (output / "orchestration-recovery.json").read_text()
    assert "initial_generator_launch" in report
    assert "permission denied" in report
    assert '"result": "blocked"' in report
    assert (output / "openhands-report.json").is_file()
    assert (output / "openhands-application.patch").is_file()
    assert (output / "generated-tests.tar.gz").is_file()


def test_controller_redacts_captured_logs_and_exceptions_in_artifact(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    scripts = trusted / ".github/scripts"
    scripts.mkdir(parents=True)
    for name in ("generate_tests.py", "pr_test_harness.py", "safe_extract.py"):
        (scripts / name).write_text("# trusted baseline\n")
    secret = "runner-secret-value-123"
    automation = tmp_path / "artifacts"
    monkeypatch.setattr(
        recovery_controller,
        "_run_generator",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr=f"auth token {secret}"
        ),
    )
    monkeypatch.setattr(recovery_controller, "_normal_result_exists", lambda _: False)
    monkeypatch.setattr(
        recovery_controller,
        "_recover_staged_script",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(f"failed with {secret}")),
    )

    assert recovery_controller.run_controller(
        trusted,
        {
            "AUTOMATION_DIR": str(automation),
            "GITHUB_WORKSPACE": str(tmp_path),
            "CUSTOM_RUNNER_TOKEN": secret,
        },
    ) == 1
    report = (automation / "generated-tests/orchestration-recovery.json").read_text()
    assert secret not in report
    assert report.count("[REDACTED]") >= 2
    for name in ("openhands-report.json", "openhands-test-results.json", "orchestration-recovery.json"):
        artifact = (automation / "generated-tests" / name).read_text()
        assert secret not in artifact

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

from pr_test_harness import (
    ensure_failure_artifacts,
    files_in_unified_diff,
    final_failure_category,
    load_agent_report,
    normalize_unified_git_patch,
    run_repair_loop,
    sandbox_test_command,
    update_validation_marker,
    validate_test_command,
    verify_test_environment,
)


def _report(**fields):
    report = {
        "schema_version": 1,
        "detected_stack": {"languages": ["python"]},
        "environment": {"status": "ready"},
        "test_commands": ["python -m pytest"],
        "failures": [],
        "dependency_installation_failures": [],
        "root_causes": [],
        "modified_files": [],
        "proposed_patches": [],
        "verified_test_results": [],
        "blockers": [],
    }
    report.update(fields)
    return report


def test_agent_failure_category_requires_verbatim_output_evidence(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report(failures=[{
            "attempt": 1,
            "category": "application bug",
            "evidence": ["assert 4 == 5"],
            "root_cause": "Implementation returned the wrong value.",
        }])))
    attempts = [{"attempt": 1, "exit_code": 1, "output": "E assert 4 == 5"}]

    report = load_agent_report(path, attempts)

    assert report["failures"][0]["category"] == "application bug"
    assert report["report_validation_errors"] == []
    assert final_failure_category(report, attempts, True) == "application bug"


def test_agent_classification_is_unknown_when_evidence_does_not_match(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report(failures=[{
            "attempt": 1,
            "category": "incorrect test",
            "evidence": ["not in captured output"],
            "root_cause": "Unsupported claim.",
        }])))
    attempts = [{"attempt": 1, "exit_code": 1, "output": "assert 4 == 5"}]

    report = load_agent_report(path, attempts)

    assert report["failures"][0]["category"] == "unknown"
    assert report["report_validation_errors"]


def test_dependency_install_failure_is_reported_as_dependency(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report(
        environment={"status": "blocked"},
        dependency_installation_failures=[{
            "name": "sample-package",
            "manager": "pip",
            "command": "pip install sample-package",
            "exit_code": 1,
            "output": "Could not find a version that satisfies the requirement",
        }],
    )))

    report = load_agent_report(path, [])

    assert report["dependency_installation_failures"][0]["category"] == "dependency"
    assert final_failure_category(report, [], False) == "dependency"


def test_malformed_dependency_install_record_is_not_classified_as_dependency(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report(
        environment={"status": "blocked"},
        dependency_installation_failures=[{
            "name": "sample-package",
            "exit_code": 1,
            "output": "install failed",
        }],
    )))

    report = load_agent_report(path, [])

    assert report["dependency_installation_failures"] == []
    assert "Dependency sample-package" in report["report_validation_errors"][0]
    assert final_failure_category(report, [], False) == "environment"


def test_classification_cannot_override_a_successful_harness_run(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report(failures=[{
            "attempt": 1,
            "category": "application bug",
            "evidence": ["old failing output"],
            "root_cause": "Stale diagnosis.",
        }])))
    attempts = [{"attempt": 1, "exit_code": 0, "output": "all passed"}]

    report = load_agent_report(path, attempts)

    assert final_failure_category(report, attempts, True) is None
    assert report["failures"][0]["category"] == "unknown"


def test_unified_diff_paths_are_reported_relative_to_repository():
    patch = """--- a/old.py
+++ b/Calculator.py
@@ -1 +1 @@
-old
+new
"""

    assert files_in_unified_diff(patch) == ["Calculator.py"]
    assert files_in_unified_diff("--- a/removed.py\n+++ /dev/null\n") == ["removed.py"]


def test_generated_patch_is_accepted_by_git_apply_check(tmp_path):
    original = tmp_path / "original"
    modified = tmp_path / "modified"
    clean = tmp_path / "clean"
    for folder in (original, modified, clean):
        folder.mkdir()
    (original / "Calculator.py").write_text("def add(a, b):\n    return a - b\n")
    (original / "removed.py").write_text("remove me\n")
    shutil.copytree(original, modified, dirs_exist_ok=True)
    shutil.copytree(original, clean, dirs_exist_ok=True)
    (modified / "Calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (modified / "new file.py").write_text("def subtract(a, b):\n    return a - b\n")
    (modified / "removed.py").unlink()

    raw = subprocess.run(
        ["diff", "-ruN", str(original), str(modified)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert raw.returncode == 1
    patch = normalize_unified_git_patch(raw.stdout, str(original), str(modified))
    patch_path = tmp_path / "openhands-application.patch"
    patch_path.write_text(patch)
    subprocess.run(["git", "init", str(clean)], check=True, capture_output=True)
    checked = subprocess.run(
        ["git", "-C", str(clean), "apply", "--check", str(patch_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert checked.returncode == 0, checked.stderr


def test_exception_path_always_writes_report_and_test_archive(tmp_path):
    (tmp_path / "generated-tests.tar.gz").write_bytes(b"partial archive")
    report_path, archive_path = ensure_failure_artifacts(tmp_path, "RuntimeError")

    report = json.loads(report_path.read_text())
    assert report["result"] == "failed"
    assert report["workflow_error"] == "RuntimeError"
    assert report["compatibility"]["claim"] is None
    assert archive_path.is_file()
    import tarfile
    with tarfile.open(archive_path, "r:gz") as archive:
        assert archive.getnames() == ["tests/generated"]
    assert report["report_validation_errors"]
    assert "Execution ended before the agent report could be validated." in report["blockers"]


def test_exception_path_preserves_a_harness_validated_diagnosis(tmp_path):
    report_path = tmp_path / "openhands-report.json"
    report_path.write_text(json.dumps({
        "result": "failed",
        "report_valid": True,
        "final_failure_category": "application bug",
        "failures": [{"attempt": 1, "category": "application bug"}],
        "report_validation_errors": [],
    }))
    (tmp_path / ".report-validated").touch()

    ensure_failure_artifacts(tmp_path, "RuntimeError")

    report = json.loads(report_path.read_text())
    assert report["final_failure_category"] == "application bug"
    assert report["failures"][0]["category"] == "application bug"
    assert not (tmp_path / ".report-validated").exists()


def test_invalid_report_is_never_marked_or_preserved_on_exception(tmp_path):
    report_path = tmp_path / "openhands-report.json"
    marker = tmp_path / ".report-validated"
    report_path.write_text(json.dumps({
        "result": "failed",
        "report_valid": False,
        "final_failure_category": "application bug",
        "failures": [{"attempt": 1, "category": "application bug"}],
        "report_validation_errors": ["evidence mismatch"],
    }))

    assert update_validation_marker(marker, json.loads(report_path.read_text())) is False
    assert not marker.exists()
    ensure_failure_artifacts(tmp_path, "RuntimeError")
    report = json.loads(report_path.read_text())
    assert report["failures"] == []
    assert report["final_failure_category"] == "environment"


def test_exception_handler_rejects_stale_marker_for_invalid_report(tmp_path):
    report_path = tmp_path / "openhands-report.json"
    report_path.write_text(json.dumps({
        "result": "failed",
        "report_valid": False,
        "report_validation_errors": ["invalid evidence"],
        "final_failure_category": "application bug",
        "failures": [{"attempt": 1, "category": "application bug"}],
    }))
    (tmp_path / ".report-validated").touch()

    ensure_failure_artifacts(tmp_path, "RuntimeError")

    report = json.loads(report_path.read_text())
    assert report["failures"] == []
    assert report["final_failure_category"] == "environment"


def _python_project(root):
    (root / "Calculator.py").write_text("def add(a, b): return a + b\n")
    tests = root / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "test_calc.py").write_text("def test_add(): assert True\n")
    (root / "requirements-dev.txt").write_text("pytest>=7\n")


def test_valid_test_command_accepts_full_pytest_suite_and_quotes_safely(tmp_path):
    _python_project(tmp_path)

    command = validate_test_command(
        ".venv/bin/python -m pytest -q --tb=short", tmp_path
    )

    assert command == ".venv/bin/python -m pytest -q --tb=short"
    sandbox_command = sandbox_test_command(command)
    assert "-u SESSION_API_KEY" in sandbox_command
    assert "&&" in sandbox_command  # harness-owned workspace prefix only
    assert ";" not in sandbox_command


def test_uv_offline_commands_accept_both_pytest_entry_points(tmp_path):
    _python_project(tmp_path)

    assert validate_test_command("uv run --offline --no-sync pytest", tmp_path) == \
        "uv run --offline --no-sync pytest"
    assert validate_test_command(
        "uv run --offline --no-sync python -m pytest", tmp_path
    ) == "uv run --offline --no-sync python -m pytest"


def test_pytest_rejects_file_directory_and_selection_options(tmp_path):
    _python_project(tmp_path)
    rejected = [
        "pytest tests/test_calc.py",
        "pytest tests",
        "pytest -k add",
        "pytest -m slow",
        "pytest --ignore=tests/test_calc.py",
        "pytest --deselect=tests/test_calc.py::test_add",
        "pytest --lf",
        "python -m pytest -q -k add",
    ]

    for command in rejected:
        try:
            validate_test_command(command, tmp_path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"partial pytest selection was accepted: {command}")


def _javascript_project(root, test_script):
    (root / "package.json").write_text(json.dumps({
        "packageManager": "npm@10.0.0",
        "scripts": {"test": test_script},
    }))
    (root / "package-lock.json").write_text("{}")


def test_javascript_test_runners_reject_partial_suite_selections(tmp_path):
    cases = [
        ("jest tests/unit/foo.test.js", "npm test"),
        ("jest", "npm test -- --testNamePattern=foo"),
        ("jest", "npm test -- tests/unit/foo.test.js"),
        ("jest --shard=1/2", "npm test"),
        ("jest --testPathIgnorePatterns=integration", "npm test"),
        ("jest --bail", "npm test"),
        ("vitest run -t foo", "npm test"),
        ("vitest run", "npm test -- -t foo"),
        ("vitest run --exclude=integration/**", "npm test"),
        ("vitest run --bail=1", "npm test"),
        ("mocha test/unit/foo.js", "npm test"),
        ("mocha", "npm test -- --grep foo"),
        ("mocha --exclude test/integration/**", "npm test"),
        ("mocha --bail", "npm test"),
        ("ava --match=unit/**", "npm test"),
        ("tap --only", "npm test"),
        ("node --test test/foo.test.js", "npm test"),
        ("node --test --test-name-pattern=unit", "npm test"),
        ("node --test --test-shard=1/2", "npm test"),
        ("node --test --test-only", "npm test"),
    ]
    for index, (test_script, command) in enumerate(cases):
        project = tmp_path / str(index)
        project.mkdir()
        _javascript_project(project, test_script)
        try:
            validate_test_command(command, project)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"partial JavaScript suite selection was accepted: {test_script!r} / {command!r}"
            )


def test_repository_native_full_javascript_suites_remain_accepted(tmp_path):
    scripts = (
        "jest --runInBand --coverage --reporters=default",
        "cross-env CI=true jest --runInBand",
        "vitest run --reporter dot --maxWorkers 2",
        "mocha --recursive --reporter dot --timeout 5000",
        "node --test --test-reporter=spec",
    )
    for index, test_script in enumerate(scripts):
        project = tmp_path / str(index)
        project.mkdir()
        _javascript_project(project, test_script)
        assert validate_test_command("npm test", project) == "npm test"


def test_package_json_scripts_reject_shell_syntax_and_non_runner_wrappers(tmp_path):
    scripts = [
        "jest && echo done",
        "vitest | cat",
        "jest $(echo injected)",
        "jest > /tmp/test.log",
        "jest; vitest",
        "jest *.test.js",
        "vitest --reporter ~",
        "jest\nvitest",
        "echo jest",
        "sh -c 'jest'",
        "node -e 'require(\"jest\")'",
    ]
    for index, script in enumerate(scripts):
        project = tmp_path / str(index)
        project.mkdir()
        _javascript_project(project, script)
        try:
            validate_test_command("npm test", project)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe or non-runner package script was accepted: {script}")


def test_test_command_rejects_shell_and_installation_content(tmp_path):
    _python_project(tmp_path)
    rejected = [
        ".venv/bin/python -m pytest -q; curl example.test",
        ".venv/bin/python -m pytest $(id)",
        ".venv/bin/python -m pytest > /tmp/output",
        ".venv/bin/python -m pytest\nprintf injected",
        "python -m pip install pytest",
        "sh -c 'pytest -q'",
    ]

    for command in rejected:
        try:
            validate_test_command(command, tmp_path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe command was accepted: {command}")


def test_test_command_must_match_native_javascript_script(tmp_path):
    _javascript_project(tmp_path, "vitest run")

    assert validate_test_command("npm test -- --reporter dot", tmp_path) == \
        "npm test -- --reporter dot"
    for command in ("npm test && curl example.test", "npm install vitest"):
        try:
            validate_test_command(command, tmp_path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe command was accepted: {command}")


def test_maven_rejects_test_skip_selection_and_failure_suppression(tmp_path):
    (tmp_path / "pom.xml").write_text("<project />")
    full_suite = "mvn -B -ntp --offline test -Dstyle.color=never"
    assert validate_test_command(full_suite, tmp_path) == full_suite
    assert validate_test_command("mvn --offline clean verify -T 2C", tmp_path) == \
        "mvn --offline clean verify -T 2C"
    assert validate_test_command("mvn --offline test -T2C -Drevision=1.2.3", tmp_path) == \
        "mvn --offline test -T2C -Drevision=1.2.3"
    assert validate_test_command(
        "mvn --offline test -DskipTests=false -DfailIfNoTests=true", tmp_path
    ) == "mvn --offline test -DskipTests=false -DfailIfNoTests=true"
    rejected = [
        "mvn --offline test -DskipTests=true",
        "mvn --offline test -DskipITs=true",
        "mvn --offline test -Dmaven.test.skip=true",
        "mvn --offline test -Dtest=OneTest",
        "mvn --offline test -Dit.test=OneIT",
        "mvn --offline test -DfailIfNoTests=false",
        "mvn --offline test -DtestFailureIgnore=true",
        "mvn --offline test -Dgroups=fast",
        "mvn --offline test -DexcludedGroups=slow",
        "mvn --offline test -Dsurefire.excludes=**/*IT.java",
        "mvn --offline test -pl module-a",
        "mvn --offline test -fn",
        "mvn --offline test -Punit-only",
        "mvn --offline test -N",
        "mvn --offline test --unknown-selection-option",
        "mvn --offline test -Dcustom.test.mode=fast",
        "mvn --offline test -DtestFailureIgnore=true",
    ]
    for command in rejected:
        try:
            validate_test_command(command, tmp_path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Maven command could skip or narrow tests: {command}")


def test_gradle_rejects_test_skip_and_selection_options(tmp_path):
    (tmp_path / "build.gradle").write_text("plugins { id 'java' }")
    assert validate_test_command("gradle --offline test", tmp_path) == "gradle --offline test"
    assert validate_test_command(
        "gradle --offline --no-daemon clean test --max-workers=2", tmp_path
    ) == "gradle --offline --no-daemon clean test --max-workers=2"
    assert validate_test_command(
        "gradle --offline clean test --max-workers 2 --console plain", tmp_path
    ) == "gradle --offline clean test --max-workers 2 --console plain"
    rejected = [
        "gradle --offline test --tests '*OneTest'",
        "gradle --offline test --exclude-task test",
        "gradle --offline test -x test",
        "gradle --offline test -DskipTests=true",
        "gradle --offline test -PincludeTags=fast",
        "gradle --offline test -PignoreFailures=true",
        "gradle --offline test --project-dir subproject",
        "gradle --offline test --dry-run",
        "gradle --offline test --unknown-selection-option",
    ]
    for command in rejected:
        try:
            validate_test_command(command, tmp_path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Gradle command could skip or narrow tests: {command}")


class _VerificationWorkspace:
    def __init__(self, results):
        self.results = iter(results)
        self.commands = []

    def execute_command(self, command, timeout=600):
        self.commands.append(command)
        return next(self.results)


def _verification_result(code=0, stdout="ok", stderr=""):
    return SimpleNamespace(exit_code=code, stdout=stdout, stderr=stderr)


def test_environment_verification_ignores_agent_readiness_marker_and_checks_python(tmp_path):
    _python_project(tmp_path)
    (tmp_path / ".openhands_environment_ready").touch()
    workspace = _VerificationWorkspace([
        _verification_result(stdout="Python 3.12.1"),
        _verification_result(stdout="pytest 8.0"),
        _verification_result(stdout="available"),
        _verification_result(stdout="No broken requirements found."),
        _verification_result(stdout="collected 3 items"),
    ])

    result = verify_test_environment(workspace, tmp_path, "python -m pytest")

    assert result["ready"] is True
    assert result["ecosystem"] == "python"
    assert result["repository_manifests"]
    assert [check["name"] for check in result["checks"]] == [
        "python_runtime", "python_test_runner", "python_pip_availability",
        "python_dependency_consistency", "python_test_collection",
    ]
    assert all("openhands_environment_ready" not in command for command in workspace.commands)
    assert "--collect-only" in workspace.commands[-1]
    assert all("--collect-only" in command or "--version" in command or "pip check" in command or "find_spec" in command for command in workspace.commands)


def test_collection_success_does_not_override_python_dependency_failure(tmp_path):
    _python_project(tmp_path)
    workspace = _VerificationWorkspace([
        _verification_result(stdout="Python 3.12.1"),
        _verification_result(stdout="pytest 8.0"),
        _verification_result(stdout="available"),
        _verification_result(1, stderr="Version conflict: package-a requires package-b<2"),
        _verification_result(stdout="collected 3 items"),
    ])

    result = verify_test_environment(workspace, tmp_path, "python -m pytest")

    assert result["ready"] is False
    assert result["checks"][-1]["passed"] is True
    assert any("python_dependency_consistency" in failure and "package-b<2" in failure for failure in result["failures"])
    assert result["blockers"] == result["failures"]


def test_unittest_environment_verification_discovers_without_running_tests(tmp_path):
    _python_project(tmp_path)
    workspace = _VerificationWorkspace([
        _verification_result(stdout="Python 3.12.1"),
        _verification_result(stdout="usage: python -m unittest"),
        _verification_result(stdout="available"),
        _verification_result(stdout="No broken requirements found."),
        _verification_result(stdout="discovered 1 tests"),
    ])

    result = verify_test_environment(workspace, tmp_path, "python -m unittest")

    assert result["ready"] is True
    assert "unittest -h" in workspace.commands[1]
    assert "defaultTestLoader.discover" in workspace.commands[-1]
    assert "runTest" not in workspace.commands[-1]


def test_environment_verification_checks_javascript_dependencies_and_collection(tmp_path):
    _javascript_project(tmp_path, "jest --runInBand")
    workspace = _VerificationWorkspace([
        _verification_result(stdout="v22.0.0"),
        _verification_result(stdout="10.0.0"),
        _verification_result(stdout="jest 29.0.0"),
        _verification_result(stdout="found 12 packages"),
        _verification_result(stdout="test/a.test.js\ntest/b.test.js"),
    ])

    result = verify_test_environment(workspace, tmp_path, "npm test")

    assert result["ready"] is True
    assert result["ecosystem"] == "javascript"
    assert [check["name"] for check in result["checks"]] == [
        "javascript_runtime", "javascript_package_manager", "javascript_test_runner",
        "javascript_dependency_consistency", "javascript_test_collection",
    ]
    assert "npm test -- --listTests" in workspace.commands[-1]
    assert all("npm test" not in command or "--version" in command or "--listTests" in command for command in workspace.commands)


def test_environment_verification_compiles_java_tests_without_running_suite(tmp_path):
    (tmp_path / "pom.xml").write_text("<project />")
    workspace = _VerificationWorkspace([
        _verification_result(stdout='openjdk version "21"'),
        _verification_result(stdout="Apache Maven 3.9"),
        _verification_result(stdout="Tests are compiled"),
    ])

    result = verify_test_environment(workspace, tmp_path, "mvn --offline test")

    assert result["ready"] is True
    assert result["ecosystem"] == "java"
    assert workspace.commands[-1].endswith("mvn --offline --batch-mode test-compile")
    assert result["checks"][-1]["name"] == "java_test_collection"
    assert result["checks"][-1]["passed"] is None
    assert any("no supported non-executing test collection check" in item for item in result["limitations"])


def test_retry_loop_executes_again_after_repair():
    workspace = _FakeWorkspace([
        SimpleNamespace(exit_code=1, stdout="first failure", stderr=""),
        SimpleNamespace(exit_code=0, stdout="passed", stderr=""),
    ])
    conversation = _FakeConversation()

    final, attempts = run_repair_loop(workspace, conversation, "run-suite", 2)

    assert final.exit_code == 0
    assert [call[0] for call in workspace.calls] == ["run-suite", "run-suite"]
    assert len(conversation.messages) == 1
    assert [entry["exit_code"] for entry in attempts] == [1, 0]


def test_retry_loop_clamps_repairs_and_runs_final_diagnosis():
    workspace = _FakeWorkspace([
        SimpleNamespace(exit_code=1, stdout=f"failure {number}", stderr="")
        for number in range(4)
    ])
    conversation = _FakeConversation()

    final, attempts = run_repair_loop(workspace, conversation, "run-suite", 20)

    assert final.exit_code == 1
    assert len(workspace.calls) == 4  # initial run plus at most three repairs
    assert len(attempts) == 4
    assert conversation.runs == 4  # three repair turns and final diagnosis only


class _FakeWorkspace:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def execute_command(self, command, timeout):
        self.calls.append((command, timeout))
        return next(self.results)


class _FakeConversation:
    def __init__(self):
        self.messages = []
        self.runs = 0

    def send_message(self, message):
        self.messages.append(message)

    def run(self):
        self.runs += 1

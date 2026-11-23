"""Security tests for untrusted-workspace isolation from the private corpus."""

import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

import generate_tests as gt  # noqa: E402

GENERATOR_SOURCE = (
    Path(__file__).resolve().parents[1] / ".github/scripts/generate_tests.py"
).read_text(encoding="utf-8")


class _RecordingWorkspace:
    def __init__(self):
        self.uploads = []
        self.commands = []

    def file_upload(self, source_path, destination_path):
        self.uploads.append((Path(source_path).name, destination_path))
        return SimpleNamespace(success=True, error="")

    def execute_command(self, command, timeout=None):
        self.commands.append(command)
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


def _write_archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for info, payload in entries:
            archive.addfile(info, io.BytesIO(payload) if payload is not None else None)


def _file_member(name, payload=b"def test_ok(): pass\n"):
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    info.mode = 0o644
    info.size = len(payload)
    return info


def _dir_member(name="tests/regression/"):
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    info.size = 0
    return info


def test_untrusted_forward_env_contains_no_secrets():
    assert "SESSION_API_KEY" in gt.UNTRUSTED_FORWARD_ENV
    assert "OH_SESSION_API_KEYS_0" in gt.UNTRUSTED_FORWARD_ENV
    for secret in (
        "LLM_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "REGRESSION_TESTS_TOKEN",
    ):
        assert secret not in gt.UNTRUSTED_FORWARD_ENV


def test_upload_pr_source_only_sends_pr_archive_and_extractor(tmp_path):
    workspace = _RecordingWorkspace()
    archive = tmp_path / "pr-repository.tar.gz"
    archive.write_bytes(b"archive")
    automation_dir = tmp_path / "trusted"
    (automation_dir / ".github/scripts").mkdir(parents=True)

    gt.upload_pr_source_to_workspace(workspace, archive, automation_dir)

    assert workspace.uploads == [
        ("pr-repository.tar.gz", "/workspace/pr-repository.tar.gz"),
        ("safe_extract.py", "/workspace/safe_extract.py"),
    ]
    assert workspace.commands == ["python /workspace/safe_extract.py"]
    for source_name, destination in workspace.uploads:
        assert "regression" not in source_name.lower()
        assert "regression" not in destination.lower()
    assert all("tests/regression" not in command for command in workspace.commands)


def test_generator_never_uploads_corpus_or_helper_into_sandbox():
    forbidden = (
        'destination_path="/workspace/approved-regression-tests.tar.gz"',
        'destination_path="/workspace/safe_extract_regression_tests.py"',
        "python /workspace/safe_extract_regression_tests.py",
    )
    for snippet in forbidden:
        assert snippet not in GENERATOR_SOURCE


def test_generation_prompt_states_corpus_is_not_executed_against_pr_code():
    assert "outside this untrusted workspace" in GENERATOR_SOURCE
    assert "executed against PR code" in GENERATOR_SOURCE
    assert "structure-validated on trusted host" in GENERATOR_SOURCE


def test_corpus_validation_compiles_python_without_executing_it(tmp_path):
    marker = tmp_path / "EXECUTED_MARKER"
    source = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "def test_ok(): pass\n"
    )
    archive = tmp_path / "corpus.tar.gz"
    _write_archive(archive, [
        (_dir_member(), None),
        (_file_member("tests/regression/PR-20/test_private.py", source.encode()), source.encode()),
    ])

    result = gt.validate_regression_corpus_archive(archive)

    assert result["valid"] is True, result["errors"]
    assert result["python_files_compiled"] == 1
    assert result["executed_against_pr_code"] is False
    assert not marker.exists()


@pytest.mark.parametrize(
    ("payload", "name", "expected"),
    [
        (b"def broken(:", "tests/regression/PR-20/broken.py", "SyntaxError"),
        (b"x", "tests/regression/../../evil.py", "Unsafe regression path"),
        (b"", "tests/regression/PR-20/empty.py", "Empty regression test"),
        (b"\x00bad", "tests/regression/PR-20/binary.py", "Binary regression test"),
    ],
)
def test_corpus_validation_rejects_invalid_members(tmp_path, payload, name, expected):
    archive = tmp_path / "corpus.tar.gz"
    _write_archive(archive, [
        (_dir_member(), None),
        (_file_member(name, payload), payload),
    ])

    result = gt.validate_regression_corpus_archive(archive)

    assert result["valid"] is False
    assert any(expected in error for error in result["errors"]), result["errors"]


def test_corpus_validation_requires_root_and_regular_archive(tmp_path):
    empty = tmp_path / "empty.tar.gz"
    _write_archive(empty, [])
    assert gt.validate_regression_corpus_archive(empty)["valid"] is False

    no_root = tmp_path / "no-root.tar.gz"
    _write_archive(no_root, [
        (_file_member("tests/regression/PR-20/test_ok.py"), b"def test_ok(): pass\n"),
    ])
    result = gt.validate_regression_corpus_archive(no_root)
    assert result["valid"] is False
    assert any("no root" in error for error in result["errors"])

    corrupt = tmp_path / "corrupt.tar.gz"
    corrupt.write_bytes(b"not a tar")
    result = gt.validate_regression_corpus_archive(corrupt)
    assert result["valid"] is False
    assert any("ReadError" in error for error in result["errors"])


def test_corpus_validation_rejects_symlinked_archive(tmp_path):
    archive = tmp_path / "corpus.tar.gz"
    _write_archive(archive, [
        (_dir_member(), None),
        (_file_member("tests/regression/PR-20/test_ok.py"), b"def test_ok(): pass\n"),
    ])
    link = tmp_path / "link.tar.gz"
    link.symlink_to(archive)

    result = gt.validate_regression_corpus_archive(link)

    assert result["valid"] is False
    assert any("missing or unsafe" in error for error in result["errors"])


def test_full_generator_pipeline_never_sends_private_corpus_to_sandbox(
    tmp_path, monkeypatch
):
    """Exercise _main() end-to-end with a recording workspace."""
    automation = tmp_path / "trusted"
    source = tmp_path / "source"
    automation.mkdir()
    source.mkdir()
    scripts = automation / ".github/scripts"
    scripts.mkdir(parents=True)
    (scripts / "safe_extract.py").write_text("# trusted extractor\n", encoding="utf-8")

    regression_input = automation / "regression-input"
    regression_input.mkdir()
    corpus_archive = regression_input / "approved-regression-tests.tar.gz"
    with tarfile.open(corpus_archive, "w:gz") as archive:
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
    gh_dummy = "ghp_DO_NOT_USE_in_test"
    regression_dummy = "regression_DO_NOT_USE_in_test"

    env = {
        "AUTOMATION_DIR": str(automation),
        "OH_SANDBOX_NETWORK": "oh-pr-sandbox-test",
        "PR_SOURCE_DIR": str(source),
        "PR_DIFF_FILE": str(diff_file),
        "PR_NUMBER": "20",
        "PR_BASE_SHA": "base",
        "PR_HEAD_SHA": "head",
        "LLM_API_KEY": "llm_DO_NOT_USE_in_test",
        "GITHUB_STEP_SUMMARY": str(summary_path),
        "GH_TOKEN": gh_dummy,
        "REGRESSION_TESTS_TOKEN": regression_dummy,
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    agent_report = {
        "schema_version": 1,
        "detected_stack": {"languages": ["python"]},
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
        "exit_code": 0,
        "output": "1 passed",
    }]

    uploads = []
    commands = []
    workspace_kwargs = {}

    class RecordingWorkspace:
        def execute_command(self, command, timeout=600):
            commands.append(command)
            if "wc -c < /workspace/openhands-test-command" in command:
                return SimpleNamespace(exit_code=0, stdout="8", stderr="")
            if "cat -- /workspace/openhands-test-command" in command:
                return SimpleNamespace(exit_code=0, stdout="pytest -q", stderr="")
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        def file_upload(self, source_path, destination_path):
            uploads.append((str(source_path), str(destination_path)))
            return SimpleNamespace(success=True, error=None)

        def file_download(self, source_path, destination_path):
            if "openhands-report.json" in source_path:
                content = json.dumps(agent_report)
            else:
                content = ""
            Path(destination_path).write_text(content, encoding="utf-8")
            return SimpleNamespace(success=True, error=None)

        def close(self):
            return None

    class RecordingDockerWorkspace:
        def __init__(self, **kwargs):
            workspace_kwargs.update(kwargs)
            self.workspace = RecordingWorkspace()

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
    monkeypatch.setattr(gt, "validate_test_command", lambda *a, **k: "pytest -q")
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
    monkeypatch.setattr(gt, "sandbox_test_command", lambda *a, **k: "pytest -q")
    monkeypatch.setattr(
        gt,
        "run_repair_loop",
        lambda *a, **k: (
            SimpleNamespace(exit_code=0, stdout="1 passed", stderr=""),
            attempt_records,
        ),
    )

    gt._main()

    # Only the PR source archive and the trusted extractor are uploaded.
    assert [Path(source_path).name for source_path, _ in uploads] == [
        "pr-repository.tar.gz",
        "safe_extract.py",
    ]
    assert [destination for _, destination in uploads] == [
        "/workspace/pr-repository.tar.gz",
        "/workspace/safe_extract.py",
    ]
    for source_path, destination in uploads:
        assert "regression" not in source_path.lower()
        assert "approved" not in source_path.lower()
        assert "regression" not in destination.lower()
        assert regression_dummy not in source_path
        assert gh_dummy not in source_path

    # No recorded command exposes or extracts private regression data.
    for command in commands:
        lowered = command.lower()
        assert "regression" not in lowered
        assert "approved" not in lowered
        assert "token" not in lowered
        assert "secret" not in lowered
        assert gh_dummy not in command
        assert regression_dummy not in command

    # Constructor kwargs do not forward GitHub or regression credentials.
    assert workspace_kwargs["forward_env"] == gt.UNTRUSTED_FORWARD_ENV
    forwarded = " ".join(str(value) for value in workspace_kwargs.values())
    assert gh_dummy not in forwarded
    assert regression_dummy not in forwarded
    assert "GITHUB_TOKEN" not in forwarded
    assert "REGRESSION_TESTS_TOKEN" not in forwarded

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

from pr_test_harness import detect_python_manifests


def test_detect_pip_project():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "requirements.txt").write_text("pytest>=7.0\n", encoding="utf-8")
        env = detect_python_manifests(root, runtime_version="3.12.3")
        assert env["package_manager"] == "pip"
        assert env["is_supported"] is True
        assert env["unsupported_reason"] is None
        assert env["declared_python_version"] is None
        assert "Project does not declare a Python version constraint" in env["python_version_assumption"]
        assert env["effective_python_version"] == "3.12.3"
        assert "requirements.txt" in env["manifest_files"]
        assert env["lockfile_policy"] == "preserve_manifest_pins"


def test_detect_python_version_from_dotfile():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "requirements.txt").write_text("pytest\n", encoding="utf-8")
        (root / ".python-version").write_text("3.11.4\n", encoding="utf-8")
        env = detect_python_manifests(root, runtime_version="3.12.3")
        assert env["declared_python_version"] == "3.11.4"
        assert env["python_version_assumption"] is None
        assert env["effective_python_version"] == "3.11.4"


def test_detect_python_version_from_tool_versions():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "requirements.txt").write_text("pytest\n", encoding="utf-8")
        (root / ".tool-versions").write_text("nodejs 20.0.0\npython 3.10.8\n", encoding="utf-8")
        env = detect_python_manifests(root, runtime_version="3.12.3")
        assert env["declared_python_version"] == "3.10.8"
        assert env["python_version_assumption"] is None


def test_detect_poetry_with_lockfile():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pyproject.toml").write_text(
            '[tool.poetry]\nname = "demo"\n[tool.poetry.dependencies]\npython = "^3.11"\n',
            encoding="utf-8",
        )
        (root / "poetry.lock").write_text('[[package]]\nname = "pytest"\n', encoding="utf-8")
        env = detect_python_manifests(root)
        assert env["package_manager"] == "poetry"
        assert env["is_supported"] is True
        assert env["lockfile_present"] is True
        assert env["lockfile_policy"] == "enforce_pinned"
        assert env["declared_python_version"] == "^3.11"


def test_detect_poetry_missing_lockfile_requires_policy():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pyproject.toml").write_text(
            '[tool.poetry]\nname = "demo"\n[tool.poetry.dependencies]\npython = "^3.11"\n',
            encoding="utf-8",
        )
        env = detect_python_manifests(root)
        assert env["package_manager"] == "poetry"
        assert env["is_supported"] is True
        assert env["lockfile_present"] is False
        assert env["lockfile_policy"] == "missing_lockfile_requires_policy"


def test_detect_pipenv_with_lockfile():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "Pipfile").write_text('[packages]\nrequests = "*"\n', encoding="utf-8")
        (root / "Pipfile.lock").write_text('{"_meta": {"requires": {"python_version": "3.11"}}}', encoding="utf-8")
        env = detect_python_manifests(root)
        assert env["package_manager"] == "pipenv"
        assert env["is_supported"] is True
        assert env["lockfile_present"] is True
        assert env["lockfile_policy"] == "enforce_pinned"
        assert env["declared_python_version"] == "3.11"


def test_detect_pipenv_missing_lockfile_requires_policy():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "Pipfile").write_text('[packages]\nrequests = "*"\n', encoding="utf-8")
        env = detect_python_manifests(root)
        assert env["package_manager"] == "pipenv"
        assert env["is_supported"] is True
        assert env["lockfile_present"] is False
        assert env["lockfile_policy"] == "missing_lockfile_requires_policy"


def test_detect_uv_with_and_without_lockfile():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pyproject.toml").write_text('[tool.uv]\ndev-dependencies = ["pytest"]\n', encoding="utf-8")
        env_no_lock = detect_python_manifests(root)
        assert env_no_lock["package_manager"] == "uv"
        assert env_no_lock["lockfile_present"] is False
        assert env_no_lock["lockfile_policy"] == "missing_lockfile_requires_policy"

        (root / "uv.lock").write_text('[[package]]\nname = "pytest"\n', encoding="utf-8")
        env_with_lock = detect_python_manifests(root)
        assert env_with_lock["package_manager"] == "uv"
        assert env_with_lock["lockfile_present"] is True
        assert env_with_lock["lockfile_policy"] == "enforce_pinned"


def test_detect_unsupported_conda():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "environment.yml").write_text("name: demo\n", encoding="utf-8")
        env = detect_python_manifests(root)
        assert env["package_manager"] == "conda"
        assert env["is_supported"] is False
        assert "Detected 'conda' package manager, which is not supported" in env["unsupported_reason"]
        assert env["lockfile_policy"] == "unsupported_manager"


def test_detect_unsupported_pdm():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pdm.lock").write_text("", encoding="utf-8")
        env = detect_python_manifests(root)
        assert env["package_manager"] == "pdm"
        assert env["is_supported"] is False
        assert "Detected 'pdm' package manager, which is not supported" in env["unsupported_reason"]


def test_detect_unsupported_pixi():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pixi.toml").write_text("", encoding="utf-8")
        env = detect_python_manifests(root)
        assert env["package_manager"] == "pixi"
        assert env["is_supported"] is False
        assert "Detected 'pixi' package manager, which is not supported" in env["unsupported_reason"]


def test_detect_no_manifest():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        env = detect_python_manifests(root)
        assert env["package_manager"] is None
        assert env["is_supported"] is False
        assert "No recognized Python package manager manifest found" in env["unsupported_reason"]


def test_empty_python_version_file_does_not_crash():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "requirements.txt").write_text("pytest\n", encoding="utf-8")
        (root / ".python-version").write_text("", encoding="utf-8")
        env = detect_python_manifests(root, runtime_version="3.12.3")
        assert env["declared_python_version"] is None
        assert env["effective_python_version"] == "3.12.3"


def test_empty_tool_versions_file_does_not_crash():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "requirements.txt").write_text("pytest\n", encoding="utf-8")
        (root / ".tool-versions").write_text("", encoding="utf-8")
        env = detect_python_manifests(root, runtime_version="3.12.3")
        assert env["declared_python_version"] is None
        assert env["effective_python_version"] == "3.12.3"


def test_poetry_python_constraint_uses_available_runtime():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pyproject.toml").write_text(
            '[tool.poetry.dependencies]\npython = "^3.11"\n',
            encoding="utf-8",
        )
        env = detect_python_manifests(root, runtime_version="3.12.3")
        assert env["declared_python_version"] == "^3.11"
        assert env["effective_python_version"] == "3.12.3"
        assert env["python_version_constraint"] == "^3.11"


def test_pep621_python_constraint_uses_available_runtime():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pyproject.toml").write_text(
            '[project]\nname = "sample"\nrequires-python = ">=3.10,<4"\n',
            encoding="utf-8",
        )
        env = detect_python_manifests(root, runtime_version="3.12.3")
        assert env["declared_python_version"] == ">=3.10,<4"
        assert env["effective_python_version"] == "3.12.3"
        assert env["python_version_constraint"] == ">=3.10,<4"

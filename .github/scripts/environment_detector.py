"""Trusted repository environment detection.

Detection performed for repair decisions reads repository state **inside the
sandbox** through the trusted workspace adapter. Host-side detection remains
available for offline tests, but its result is never used for sandbox installs.
This module never executes repository code or PR-controlled commands.
"""

from __future__ import annotations

import configparser
import json
import re
import shlex
import sys
import tomllib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DETECTOR_SCHEMA_VERSION = 1
SANDBOX_ROOT = "/workspace"

_PYTEST_IMPORT_RE = re.compile(
    r"(?m)^\s*(?:import\s+pytest|from\s+pytest(?:\s|\.|\b))"
)
_UNITTEST_IMPORT_RE = re.compile(r"(?m)^\s*import\s+unittest\b")
_REQUIREMENT_RE = re.compile(r"requirements[\w.-]*\.txt\Z")
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".pytest_cache", ".venv", "venv", "env", ".env"}

_SANDBOX_SCAN_SCRIPT = r'''
import json, os
root = ROOT
skip = SKIP
relevant = RELEVANT
max_files = 250
max_file_bytes = 65536
max_total = 2097152
files = {}
total = 0
python_files = 0
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [
        name for name in dirnames
        if name not in skip and not os.path.islink(os.path.join(dirpath, name))
    ]
    for name in filenames:
        full = os.path.join(dirpath, name)
        rel = os.path.relpath(full, root)
        if os.path.islink(full):
            continue
        if name.endswith(".py"):
            python_files += 1
        selected = (
            name in relevant
            or bool(re.search(r"requirements[\w.-]*\.txt\Z", name))
            or (
                name.endswith(".py")
                and (
                    rel.startswith("tests/")
                    or rel.startswith("test/")
                    or name.startswith("test_")
                    or name.endswith("_test.py")
                )
            )
        )
        if not selected or len(files) >= max_files:
            continue
        try:
            size = os.path.getsize(full)
            if size > max_file_bytes or total + size > max_total:
                continue
            with open(full, encoding="utf-8", errors="ignore") as handle:
                text = handle.read(max_file_bytes)
        except OSError:
            continue
        files[rel] = text
        total += len(text)
print(json.dumps({"files": files, "python_files": python_files}))
'''


def _validate_root(root: str) -> str:
    if not isinstance(root, str) or not root.startswith(SANDBOX_ROOT + "/") and root != SANDBOX_ROOT:
        raise ValueError("repository root must be inside /workspace")
    parts = Path(root).parts
    if ".." in parts:
        raise ValueError("repository root must not contain traversal")
    return root.rstrip("/") or SANDBOX_ROOT


def _read_text(path: Path, limit: int = 512 * 1024) -> str:
    try:
        if path.stat().st_size > limit:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _host_file_map(repository_root: Path) -> tuple[dict, int]:
    root = Path(repository_root).resolve()
    files: dict[str, str] = {}
    python_files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if any(part in _SKIP_DIRS for part in Path(relative).parts):
            continue
        if path.suffix == ".py":
            python_files += 1
        selected = (
            Path(relative).name in {
                "pyproject.toml", "setup.py", "setup.cfg", "tox.ini", "Pipfile",
                "Pipfile.lock", "poetry.lock", "uv.lock", "constraints.txt",
                ".python-version", ".tool-versions", "environment.yml", "environment.yaml",
                "conda.yaml", "pdm.lock", "pixi.toml", "pixi.lock", "flit.ini",
                "hatch.toml",
            }
            or _REQUIREMENT_RE.match(Path(relative).name)
            or (
                path.suffix == ".py"
                and (
                    relative.startswith("tests/")
                    or relative.startswith("test/")
                    or Path(relative).name.startswith("test_")
                    or Path(relative).name.endswith("_test.py")
                )
            )
        )
        if selected and len(files) < 250:
            text = _read_text(path, 65536)
            if text:
                files[relative] = text
    return files, python_files


def _sandbox_file_map(workspace, root: str) -> tuple[dict, int]:
    """Read repository files from the sandbox using a trusted read-only script."""
    root = _validate_root(root)
    relevant = [
        "pyproject.toml", "setup.py", "setup.cfg", "tox.ini", "Pipfile",
        "Pipfile.lock", "poetry.lock", "uv.lock", "constraints.txt",
        ".python-version", ".tool-versions", "environment.yml", "environment.yaml",
        "conda.yaml", "pdm.lock", "pixi.toml", "pixi.lock", "flit.ini", "hatch.toml",
    ]
    script = (
        _SANDBOX_SCAN_SCRIPT
        .replace("ROOT", json.dumps(root))
        .replace("SKIP", json.dumps(sorted(_SKIP_DIRS)))
        .replace("RELEVANT", json.dumps(relevant))
    )
    script = "import re\n" + script
    command = None
    for executable in ("python3", "python"):
        candidate = f"{executable} -c {shlex.quote(script)}"
        result = workspace.execute_command(candidate, timeout=120)
        if result.exit_code == 0:
            command = candidate
            break
    if command is None:
        raise RuntimeError("Could not inspect repository files inside the sandbox")
    report = None
    for line in reversed((result.stdout or "").splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            report = value
            break
    if not isinstance(report, dict) or not isinstance(report.get("files"), dict):
        raise RuntimeError("Sandbox repository inspection returned no usable report")
    return report["files"], int(report.get("python_files") or 0)


def _toml(text: str) -> dict:
    try:
        value = tomllib.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _detect_test_framework_from_files(files: dict) -> str | None:
    pyproject = files.get("pyproject.toml", "")
    if re.search(r"\bpytest\b", pyproject, re.I):
        return "pytest"
    for name, text in files.items():
        if not name.endswith(".py"):
            continue
        if _PYTEST_IMPORT_RE.search(text):
            return "pytest"
        if _UNITTEST_IMPORT_RE.search(text):
            return "unittest"
    return None


def _build_detection(files: dict, python_files: int, root: str) -> dict:
    declared_python = None
    version_text = files.get(".python-version", "").strip()
    if version_text:
        declared_python = version_text.splitlines()[0].strip()

    tool_versions = files.get(".tool-versions", "")
    if declared_python is None:
        for line in tool_versions.splitlines():
            if line.strip().startswith("python "):
                declared_python = line.strip().split(maxsplit=1)[1].strip()
                break

    pyproject = _toml(files.get("pyproject.toml", ""))
    project = pyproject.get("project", {}) if isinstance(pyproject.get("project"), dict) else {}
    tools = pyproject.get("tool", {}) if isinstance(pyproject.get("tool"), dict) else {}
    if declared_python is None and isinstance(project.get("requires-python"), str):
        declared_python = project["requires-python"].strip()
    if declared_python is None and isinstance(tools.get("poetry"), dict):
        poetry_deps = tools["poetry"].get("dependencies", {})
        if isinstance(poetry_deps, dict) and isinstance(poetry_deps.get("python"), str):
            declared_python = poetry_deps["python"].strip()
    if declared_python is None and "Pipfile" in files:
        pipfile = _toml(files["Pipfile"])
        requires = pipfile.get("requires", {})
        if isinstance(requires, dict):
            declared_python = requires.get("python_version") or requires.get("python_full_version")

    setup_cfg = files.get("setup.cfg", "")
    if declared_python is None and setup_cfg:
        try:
            cfg = configparser.ConfigParser()
            cfg.read_string(setup_cfg)
            if cfg.has_option("options", "python_requires"):
                declared_python = cfg.get("options", "python_requires").strip()
        except Exception:
            pass

    manifest_files = []
    for name in (
        "pyproject.toml", "setup.py", "setup.cfg", "tox.ini", "Pipfile",
        "constraints.txt", ".python-version", ".tool-versions",
    ):
        if name in files:
            manifest_files.append(name)
    requirements_files = sorted(
        name for name in files if _REQUIREMENT_RE.match(name)
    )
    manifest_files.extend(requirements_files)

    lockfiles = [
        name for name in ("poetry.lock", "uv.lock", "Pipfile.lock", "constraints.txt", "pdm.lock", "pixi.lock")
        if name in files
    ]

    package_manager = None
    if any(name in files for name in ("environment.yml", "environment.yaml", "conda.yaml")):
        package_manager = "conda"
    elif "pdm.lock" in files or "pdm" in tools:
        package_manager = "pdm"
    elif any(name in files for name in ("pixi.toml", "pixi.lock")) or "pixi" in tools:
        package_manager = "pixi"
    elif "flit.ini" in files or "flit" in tools:
        package_manager = "flit"
    elif "hatch.toml" in files or "hatch" in tools:
        package_manager = "hatch"
    elif "poetry.lock" in files or "poetry" in tools:
        package_manager = "poetry"
    elif any(name in files for name in ("Pipfile", "Pipfile.lock")):
        package_manager = "pipenv"
    elif "uv.lock" in files or "uv" in tools:
        package_manager = "uv"
    elif requirements_files or "setup.py" in files or "setup.cfg" in files or project:
        package_manager = "pip"

    supported = {"pip", "poetry", "uv", "pipenv"}
    unsupported = {"conda", "pdm", "pixi", "flit", "hatch"}
    is_supported = package_manager in supported
    unsupported_reason = None
    if package_manager in unsupported:
        unsupported_reason = f"Detected unsupported package manager {package_manager!r}"
    elif package_manager is None:
        unsupported_reason = "No recognized Python package manager manifest found"

    if package_manager in {"poetry", "pipenv", "uv"}:
        lockfile_present = any(
            name in files for name in {"poetry.lock", "Pipfile.lock", "uv.lock"}
        )
        lockfile_policy = "enforce_pinned" if lockfile_present else "missing_lockfile_requires_policy"
    elif package_manager == "pip":
        lockfile_present = "constraints.txt" in files
        lockfile_policy = "preserve_manifest_pins"
    else:
        lockfile_present = False
        lockfile_policy = "unsupported_manager"

    return {
        "schema_version": DETECTOR_SCHEMA_VERSION,
        "languages": ["python"] if python_files or manifest_files else [],
        "python": {
            "detected": bool(python_files or manifest_files),
            "package_manager": package_manager,
            "is_supported": is_supported,
            "unsupported_reason": unsupported_reason,
            "declared_python_version": declared_python,
            "python_version_assumption": declared_python or "No Python version declared",
            "effective_python_version": declared_python or "runtime default",
            "manifest_files": sorted(set(manifest_files)),
            "lockfiles": lockfiles,
            "lockfile_present": lockfile_present,
            "lockfile_policy": lockfile_policy,
            "test_framework": _detect_test_framework_from_files(files),
            "requirements_files": requirements_files,
            "requirement_contents": {name: files[name] for name in requirements_files},
            "pyproject_text": files.get("pyproject.toml", "")[:65536],
            "has_pyproject": "pyproject.toml" in files,
        },
        "services": [],
        "unsupported": {
            "javascript": "not implemented in Phase 1",
            "java": "not implemented in Phase 1",
            "services": "not implemented in Phase 1",
        },
        "evidence": {
            "repository_root": root,
            "python_files": python_files,
            "source": "sandbox" if root == SANDBOX_ROOT else "host",
        },
    }


def detect_environment(repository_root: Path | None = None, workspace=None) -> dict:
    """Detect repository state.

    When ``workspace`` is supplied, all repository data is read from the
    isolated sandbox at ``/workspace``. Host inspection is retained for offline
    tests only and is explicitly marked in the report.
    """
    if workspace is not None:
        files, python_files = _sandbox_file_map(workspace, SANDBOX_ROOT)
        return _build_detection(files, python_files, SANDBOX_ROOT)
    if repository_root is None:
        raise ValueError("repository_root is required for host detection")
    files, python_files = _host_file_map(Path(repository_root))
    return _build_detection(files, python_files, SANDBOX_ROOT)


def declared_test_requirements(detection: dict) -> list[str]:
    """Return every distinct pytest pin declared in repository manifests."""
    if not isinstance(detection, dict):
        return []
    python = detection.get("python") or {}
    pins: list[str] = []
    for text in (python.get("requirement_contents") or {}).values():
        for line in text.splitlines():
            match = re.match(r"\s*pytest\s*([<>=!~]=?[^\s;]+)", line, re.I)
            if match:
                pin = "pytest" + match.group(1)
                if pin not in pins:
                    pins.append(pin)
    pyproject = python.get("pyproject_text") or ""
    for match in re.finditer(r"pytest\s*([<>=!~]=?[^\s,;\"']+)", pyproject, re.I):
        pin = "pytest" + match.group(1)
        if pin not in pins:
            pins.append(pin)
    return pins


def declared_test_requirement(detection: dict) -> str:
    """Return the single declared pytest pin, or the unpinned runner name."""
    pins = declared_test_requirements(detection)
    if len(pins) == 1:
        return pins[0]
    return "pytest"

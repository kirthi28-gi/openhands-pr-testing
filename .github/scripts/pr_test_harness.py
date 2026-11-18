"""Validation and retry helpers for the OpenHands PR test runner."""

from __future__ import annotations

import configparser
import inspect
import json
import hashlib
import os
import re
import shlex
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

try:
    import tomllib
except ImportError:
    tomllib = None

try:
    from packaging.requirements import Requirement as PackagingRequirement, InvalidRequirement
except ImportError:
    PackagingRequirement = None
    InvalidRequirement = Exception


FAILURE_CATEGORIES = {
    "dependency",
    "environment",
    "build",
    "incorrect test",
    "application bug",
    "regression",
    "unknown",
}

SUPPORTED_PACKAGE_MANAGERS = {"pip", "poetry", "uv", "pipenv"}
UNSUPPORTED_PACKAGE_MANAGERS = {"conda", "pdm", "pixi", "flit", "hatch"}

SANDBOX_ENV_ALLOWLIST = {
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "PWD",
    "PYTHONPATH", "VIRTUAL_ENV", "CI", "DEBIAN_FRONTEND",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE",
}

_SHELL_OPERATOR_CHARS = ";&|<>\n\r\x00"
_SHELL_EXPANSION_CHARS = "*?[]{}~"
_FORBIDDEN_COMMANDS = {
    "apt", "apt-get", "curl", "git", "pip", "pip3", "wget",
    "brew", "dnf", "yum", "apk",
}
_JS_TEST_RUNNERS = {
    "jest", "vitest", "mocha", "ava", "tap", "node:test",
    "react-scripts", "vue-cli-service", "ng",
}

#: Default ``PATH`` used when the sandbox shell environment is not consulted.
SANDBOX_PATH_DEFAULT = (
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
#: Extra user-level bin directories where wrappers such as Poetry/Pipenv are
#: commonly installed inside agent sandboxes.
_SANDBOX_USER_BIN = ("/home/openhands/.local/bin", "/root/.local/bin")
#: Deterministic, non-secret environment shared by every sandboxed command.
SANDBOX_ENV_DEFAULTS = {
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "SHELL": "/bin/sh",
    "PWD": "/workspace",
    "TERM": "xterm",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
#: Non-identity variables inherited from the host only when safe.
#: Identity variables such as ``USER``/``VIRTUAL_ENV``/``PYTHONPATH`` are never
#: inherited; the plan supplies them or leaves them unset.
_SANDBOX_OPTIONAL_ENV = (
    "CI",
    "DEBIAN_FRONTEND",
    "LC_ALL",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
)
PLAN_SCHEMA_VERSION = 1

_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_UNSAFE_ENV_VALUE_RE = re.compile(r'["`$\\\n\r\x00]')


class EnvironmentResolutionError(RuntimeError):
    """Raised when no safe, evidence-backed test execution plan exists.

    The error carries structured evidence so the caller can persist a
    fail-safe report instead of running a command against an unknown
    interpreter.
    """

    def __init__(self, message: str, evidence: dict | None = None):
        super().__init__(message)
        self.evidence = dict(evidence or {})


def _execution_plan_digest(
    argv: tuple[str, ...],
    env: dict[str, str],
    interpreter: str | None,
    venv_root: str | None,
    framework: str,
    ecosystem: str,
    source: str,
) -> str:
    """Return a stable digest of everything that determines execution."""
    payload = json.dumps(
        {
            "argv": list(argv),
            "env": dict(sorted(env.items())),
            "interpreter": interpreter,
            "venv_root": venv_root,
            "framework": framework,
            "ecosystem": ecosystem,
            "source": source,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TestExecutionPlan:
    """The single source of truth for one repository test execution.

    A plan resolves the command shape, interpreter, virtual environment,
    sanitized environment, and framework exactly once. Verification, recovery,
    collection, and the final suite all consume the same plan, so the
    environment that is checked is the environment that runs the tests.
    """

    # Prevent pytest from trying to collect this dataclass as a test class
    # when a test module imports it.
    __test__ = False

    argv: tuple[str, ...]
    env: dict[str, str]
    interpreter: str | None
    venv_root: str | None
    framework: str
    ecosystem: str
    source: str
    evidence: dict = field(default_factory=dict)
    wrapper: tuple[str, ...] = ()
    runner: tuple[str, ...] = ()
    digest: str = ""

    def __post_init__(self):
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "env", dict(self.env))
        object.__setattr__(self, "wrapper", tuple(self.wrapper))
        object.__setattr__(self, "runner", tuple(self.runner))
        if not self.argv or not all(
            isinstance(item, str) and item for item in self.argv
        ):
            raise ValueError("execution plan argv must contain non-empty strings")
        computed = _execution_plan_digest(
            self.argv,
            self.env,
            self.interpreter,
            self.venv_root,
            self.framework,
            self.ecosystem,
            self.source,
        )
        if self.digest and self.digest != computed:
            raise ValueError("execution plan digest does not match its contents")
        if not self.digest:
            object.__setattr__(self, "digest", computed)

    @classmethod
    def from_dict(cls, data: dict) -> "TestExecutionPlan":
        """Rehydrate a plan previously persisted in a report artifact."""
        if not isinstance(data, dict):
            raise ValueError("execution plan must be an object")
        required = ("argv", "env", "framework", "ecosystem", "source")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"execution plan is missing keys: {', '.join(missing)}")
        return cls(
            argv=tuple(data["argv"]),
            env=dict(data["env"]),
            interpreter=data.get("interpreter"),
            venv_root=data.get("venv_root"),
            framework=data["framework"],
            ecosystem=data["ecosystem"],
            source=data["source"],
            evidence=dict(data.get("evidence") or {}),
            wrapper=tuple(data.get("wrapper") or ()),
            runner=tuple(data.get("runner") or ()),
            digest=str(data.get("digest") or ""),
        )

    def to_dict(self) -> dict:
        """Serialize the plan without secrets (the plan never contains any)."""
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "argv": list(self.argv),
            "env": dict(sorted(self.env.items())),
            "interpreter": self.interpreter,
            "venv_root": self.venv_root,
            "framework": self.framework,
            "ecosystem": self.ecosystem,
            "source": self.source,
            "wrapper": list(self.wrapper),
            "runner": list(self.runner),
            "evidence": self.evidence,
            "digest": self.digest,
        }


def detect_python_manifests(
    repository_root: Path,
    runtime_version: str | None = None,
) -> dict:
    """Detect Python version declarations, manifests, lockfiles, and package manager.

    Does not guess or assume default Python 3.12 when undeclared; records the
    runtime assumption explicitly. Reports unsupported package managers
    clearly without guessing installation commands.
    """
    current_runtime = (
        runtime_version
        if runtime_version is not None
        else f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    manifest_files: list[str] = []
    lockfiles: list[str] = []
    declared_python: str | None = None

    # Dotfiles version detection
    pyver = repository_root / ".python-version"
    if pyver.is_file():
        manifest_files.append(".python-version")
        try:
            declared_python = pyver.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        except Exception:
            pass

    toolver = repository_root / ".tool-versions"
    if toolver.is_file():
        manifest_files.append(".tool-versions")
        if declared_python is None:
            try:
                for line in toolver.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("python "):
                        declared_python = line.strip().split(maxsplit=1)[1].strip()
                        break
            except Exception:
                pass

    # pyproject.toml
    pyproject_file = repository_root / "pyproject.toml"
    pyproject_data: dict = {}
    if pyproject_file.is_file():
        manifest_files.append("pyproject.toml")
        if tomllib is not None:
            try:
                pyproject_data = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
                if declared_python is None:
                    proj = pyproject_data.get("project", {})
                    if isinstance(proj, dict) and "requires-python" in proj:
                        declared_python = str(proj["requires-python"]).strip()
                    poetry = pyproject_data.get("tool", {}).get("poetry", {})
                    if declared_python is None and isinstance(poetry, dict):
                        p_deps = poetry.get("dependencies", {})
                        if isinstance(p_deps, dict) and "python" in p_deps:
                            declared_python = str(p_deps["python"]).strip()
            except Exception:
                pass

    # Pipfile & Pipfile.lock
    pipfile = repository_root / "Pipfile"
    if pipfile.is_file():
        manifest_files.append("Pipfile")
        if declared_python is None and tomllib is not None:
            try:
                p_data = tomllib.loads(pipfile.read_text(encoding="utf-8"))
                req = p_data.get("requires", {})
                if isinstance(req, dict):
                    declared_python = req.get("python_version") or req.get("python_full_version")
            except Exception:
                pass

    pipfile_lock = repository_root / "Pipfile.lock"
    if pipfile_lock.is_file():
        lockfiles.append("Pipfile.lock")
        if declared_python is None:
            try:
                lock_data = json.loads(pipfile_lock.read_text(encoding="utf-8"))
                req = lock_data.get("_meta", {}).get("requires", {})
                declared_python = req.get("python_version") or req.get("python_full_version")
            except Exception:
                pass

    # poetry.lock
    if (repository_root / "poetry.lock").is_file():
        lockfiles.append("poetry.lock")

    # uv.lock
    if (repository_root / "uv.lock").is_file():
        lockfiles.append("uv.lock")

    # setup.cfg
    setup_cfg = repository_root / "setup.cfg"
    if setup_cfg.is_file():
        manifest_files.append("setup.cfg")
        if declared_python is None:
            try:
                cfg = configparser.ConfigParser()
                cfg.read_string(setup_cfg.read_text(encoding="utf-8"))
                if cfg.has_option("options", "python_requires"):
                    declared_python = cfg.get("options", "python_requires").strip()
            except Exception:
                pass

    # setup.py
    if (repository_root / "setup.py").is_file():
        manifest_files.append("setup.py")

    # requirements*.txt
    for req in sorted(repository_root.glob("requirements*.txt")):
        if req.is_file():
            manifest_files.append(req.name)

    # constraints.txt
    if (repository_root / "constraints.txt").is_file():
        lockfiles.append("constraints.txt")

    # Version assumption
    python_version_constraint = None
    is_exact_python_version = (
        declared_python is not None
        and re.fullmatch(r"\d+\.\d+(?:\.\d+)?", declared_python) is not None
    )

    if is_exact_python_version:
        python_version_assumption = None
        effective_python = declared_python
    elif declared_python is not None:
        python_version_constraint = declared_python
        effective_python = current_runtime
        python_version_assumption = (
            f"Project declares Python constraint {declared_python}; "
            f"using available runtime {current_runtime}. Compatibility must be verified."
        )
    else:
        python_version_assumption = (
            f"Project does not declare a Python version constraint; using available runtime {current_runtime}"
        )
        effective_python = current_runtime

    # Package manager detection
    tools = pyproject_data.get("tool", {}) if isinstance(pyproject_data.get("tool"), dict) else {}
    backend = str(pyproject_data.get("build-system", {}).get("build-backend", ""))

    package_manager: str | None = None
    if any((repository_root / name).is_file() for name in ("environment.yml", "environment.yaml", "conda.yaml")):
        package_manager = "conda"
    elif (repository_root / "pdm.lock").is_file() or "pdm" in tools:
        package_manager = "pdm"
    elif any((repository_root / name).is_file() for name in ("pixi.toml", "pixi.lock")) or "pixi" in tools:
        package_manager = "pixi"
    elif (repository_root / "flit.ini").is_file() or "flit" in tools or "flit_core" in backend:
        package_manager = "flit"
    elif (repository_root / "hatch.toml").is_file() or "hatch" in tools or "hatchling" in backend:
        package_manager = "hatch"

    if package_manager is None:
        if (repository_root / "poetry.lock").is_file() or "poetry" in tools:
            package_manager = "poetry"
        elif (repository_root / "Pipfile").is_file() or (repository_root / "Pipfile.lock").is_file():
            package_manager = "pipenv"
        elif (repository_root / "uv.lock").is_file() or "uv" in tools:
            package_manager = "uv"
        elif any(f.startswith("requirements") for f in manifest_files) or "setup.py" in manifest_files or "setup.cfg" in manifest_files or (pyproject_file.is_file() and "project" in pyproject_data):
            package_manager = "pip"

    if package_manager in UNSUPPORTED_PACKAGE_MANAGERS:
        is_supported = False
        unsupported_reason = (
            f"Detected '{package_manager}' package manager, which is not supported. "
            f"Supported managers are: {', '.join(sorted(SUPPORTED_PACKAGE_MANAGERS))}."
        )
    elif package_manager in SUPPORTED_PACKAGE_MANAGERS:
        is_supported = True
        unsupported_reason = None
    else:
        is_supported = False
        unsupported_reason = "No recognized Python package manager manifest found."

    # Lockfile policy
    if package_manager == "poetry":
        has_lock = (repository_root / "poetry.lock").is_file()
        lockfile_present = has_lock
        lockfile_policy = "enforce_pinned" if has_lock else "missing_lockfile_requires_policy"
    elif package_manager == "pipenv":
        has_lock = (repository_root / "Pipfile.lock").is_file()
        lockfile_present = has_lock
        lockfile_policy = "enforce_pinned" if has_lock else "missing_lockfile_requires_policy"
    elif package_manager == "uv":
        has_lock = (repository_root / "uv.lock").is_file()
        lockfile_present = has_lock
        lockfile_policy = "enforce_pinned" if has_lock else "missing_lockfile_requires_policy"
    elif package_manager == "pip":
        has_lock = any((repository_root / name).is_file() for name in ("constraints.txt", "requirements-lock.txt"))
        lockfile_present = True if has_lock else None
        lockfile_policy = "preserve_manifest_pins"
    else:
        lockfile_present = False
        lockfile_policy = "unsupported_manager"

    return {
        "package_manager": package_manager,
        "is_supported": is_supported,
        "unsupported_reason": unsupported_reason,
        "declared_python_version": declared_python,
        "python_version_constraint": python_version_constraint,
        "runtime_python_version": current_runtime,
        "python_version_assumption": python_version_assumption,
        "effective_python_version": effective_python,
        "manifest_files": sorted(list(set(manifest_files))),
        "lockfiles": sorted(list(set(lockfiles))),
        "lockfile_present": lockfile_present,
        "lockfile_policy": lockfile_policy,
    }



def _lex_one_command(command: str) -> list[str]:
    if not isinstance(command, str) or not command.strip() or len(command) > 2048:
        raise ValueError("test command must be a non-empty string of at most 2048 characters")
    if command.endswith("\r\n"):
        command = command[:-2]
    elif command.endswith("\n"):
        command = command[:-1]
    if (
        any(char in command for char in _SHELL_OPERATOR_CHARS)
        or any(char in command for char in _SHELL_EXPANSION_CHARS)
        or "$" in command
        or "`" in command
    ):
        raise ValueError("shell operators, substitutions, redirections, and multiple lines are forbidden")
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError as exc:
        raise ValueError("malformed quoting in test command") from exc
    if not tokens or any(token and all(char in ";&|<>" for char in token) for token in tokens):
        raise ValueError("test command must contain one command with no shell operators")
    if any(Path(token).name.lower() in _FORBIDDEN_COMMANDS for token in tokens):
        raise ValueError("installation, network, and package-management commands are forbidden")
    return tokens


def _venv_runner_kind(token: str) -> str | None:
    """Classify a venv interpreter or pytest path token without hardcoding a location.

    Accepts sandbox-relative paths such as ``venv/bin/python`` and absolute
    paths inside the sandbox workspace (``/workspace/...``) only, and rejects
    path traversal.
    """
    parts = PurePosixPath(token).parts
    if len(parts) < 3 or ".." in parts or parts[-2] != "bin":
        return None
    if token.startswith("/"):
        # PurePosixPath keeps the root as its first component, so workspace
        # absolute paths are ('/', 'workspace', ...). Anything else is outside
        # the sandbox workspace and must not be treated as a project runner.
        if parts[:2] != ("/", "workspace"):
            return None
    if parts[-1] == "pytest":
        return "pytest"
    if re.fullmatch(r"python(?:3(?:\.\d+)?)?", parts[-1]):
        return "python"
    return None


def _interpreter_sibling_python(token: str) -> str:
    """Map a venv pytest path (``venv/bin/pytest``) to its environment interpreter."""
    parts = PurePosixPath(token).parts
    return str(PurePosixPath(*parts[:-1], "python"))


def _python_runner(tokens: list[str]) -> bool:
    if not tokens:
        return False
    kind = _venv_runner_kind(tokens[0])
    if tokens[0] == "pytest" or kind == "pytest":
        return _full_pytest_arguments(tokens[1:])
    if (
        tokens[0] in {"python", "python3"}
        or kind == "python"
        or re.fullmatch(r"python3\.\d+", tokens[0])
    ):
        return _python_module_runner(tokens[1:])
    if tokens[0] == "uv":
        if len(tokens) < 4 or tokens[1] != "run":
            return False
        options = tokens[2:]
        if options[:2] == ["--offline", "--no-sync"]:
            runner = options[2:]
        elif options[:2] == ["--no-sync", "--offline"]:
            runner = options[2:]
        else:
            return False
        if runner and runner[0] == "--":
            runner = runner[1:]
        return _python_runner(runner)
    if tokens[0] == "poetry":
        return len(tokens) >= 3 and tokens[1] == "run" and _python_runner(tokens[2:])
    if tokens[0] == "pipenv":
        return len(tokens) >= 3 and tokens[1] == "run" and _python_runner(tokens[2:])
    return False


_PYTEST_VALUE_OPTIONS = {"--tb", "--color", "--capture", "--show-capture", "--durations"}
_PYTEST_SWITCHES = {
    "-q", "--quiet", "-v", "--verbose", "-s", "--capture=no",
    "--disable-warnings", "--strict-config", "--strict-markers",
    "--no-header", "--no-summary", "--full-trace",
}


def _full_pytest_arguments(arguments: list[str]) -> bool:
    """Permit display-only pytest options; reject paths and suite selectors."""
    index = 0
    while index < len(arguments):
        option = arguments[index]
        if option in _PYTEST_SWITCHES:
            index += 1
            continue
        if option in _PYTEST_VALUE_OPTIONS:
            if index + 1 >= len(arguments):
                return False
            index += 2
            continue
        if any(option.startswith(name + "=") for name in _PYTEST_VALUE_OPTIONS):
            index += 1
            continue
        return False
    return True


def _python_module_runner(tokens: list[str]) -> bool:
    if len(tokens) < 2 or tokens[0] != "-m":
        return False
    if tokens[1] == "pytest":
        return _full_pytest_arguments(tokens[2:])
    # unittest's positional names, discovery paths, and patterns all select
    # only part of the suite. Its no-argument default discovers the suite.
    return tokens[1] == "unittest" and tokens[2:] in ([], ["-v"], ["--verbose"])


def _javascript_runner(tokens: list[str]) -> bool:
    if not tokens:
        return False
    name = Path(tokens[0]).name
    if name in _JS_TEST_RUNNERS and name not in {"react-scripts", "vue-cli-service", "ng"}:
        args = tokens[1:]
        # File arguments and runner-specific filter options silently narrow
        # the suite. Accept only common reporting/display options.
        forbidden = {
            "-t", "--testNamePattern", "--testPathPattern", "--runTestsByPath",
            "--selectProjects", "--ignoreProjects", "--onlyChanged", "--lastCommit",
            "--findRelatedTests", "-g", "--grep", "--fgrep", "--exclude",
            "--dir", "--project", "--shard", "--changedSince", "--passWithNoTests",
            "--include", "--excludeTask", "--grep", "--testPathIgnorePatterns",
            "-m", "--match", "--testPathPatterns", "--only",
            "--onlyFailures", "--ignore", "--fail-fast", "--bail",
            "--maxFailures", "--max-failures",
        }
        if any(
            arg in forbidden
            or arg.startswith(("-t=", "-g=", "-m="))
            or any(arg.startswith(flag + "=") for flag in forbidden if flag.startswith("--"))
            for arg in args
        ):
            return False
        value_options = {
            "--reporter", "--reporters", "--coverageReporters", "--outputFile",
            "--maxWorkers", "--minWorkers", "--coverageDirectory",
            "--testTimeout", "--timeout", "--slow", "--retries", "--jobs",
        }
        switches = {
            "run", "--run", "--coverage", "--no-coverage", "--watch=false",
            "--watchAll=false", "--runInBand", "--ci", "--silent", "--verbose",
            "--recursive", "--exit", "--detectOpenHandles", "--forceExit",
            "--logHeapUsage",
        }
        index = 0
        while index < len(args):
            arg = args[index]
            if arg in switches:
                index += 1
            elif arg in value_options and index + 1 < len(args):
                index += 2
            elif any(arg.startswith(flag + "=") for flag in value_options):
                index += 1
            else:
                # Includes positional file/name filters and unknown flags.
                return False
        return True
    if name == "node":
        if not tokens[1:] or tokens[1] != "--test":
            return False
        options = tokens[2:]
        value_options = {"--test-reporter", "--test-reporter-destination"}
        index = 0
        while index < len(options):
            option = options[index]
            if option in value_options and index + 1 < len(options):
                index += 2
            elif any(option.startswith(name + "=") for name in value_options):
                index += 1
            elif option == "--experimental-test-coverage":
                index += 1
            else:
                # Includes test files, --test-name-pattern, sharding, and
                # --test-only selections.
                return False
        return True
    if name == "react-scripts":
        return len(tokens) > 1 and tokens[1] == "test" and _javascript_runner(["jest", *tokens[2:]])
    if name == "vue-cli-service":
        return len(tokens) > 1 and tokens[1] == "test:unit" and _javascript_runner(["mocha", *tokens[2:]])
    if name == "ng":
        return len(tokens) > 1 and tokens[1] == "test" and _javascript_runner(["jest", *tokens[2:]])
    if name == "cross-env":
        index = 1
        while index < len(tokens) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[index]):
            index += 1
        return index < len(tokens) and _javascript_runner(tokens[index:])
    return False


def validate_test_command(command: str, repository_root: Path) -> str:
    """Accept only one safe command matching a runner configured by the repository."""
    tokens = _lex_one_command(command)
    root = repository_root.resolve()
    token0 = Path(tokens[0]).name

    python_sources = any(root.rglob("*.py"))
    python_tests = any(
        candidate.is_file()
        for folder in (root / "tests", root / "test")
        if folder.is_dir()
        for candidate in folder.rglob("*.py")
    ) or any(root.glob("test_*.py"))
    pyproject_text = (root / "pyproject.toml").read_text(errors="ignore") if (root / "pyproject.toml").is_file() else ""
    python_manifests = list(root.glob("requirements*.txt")) + [
        path for path in (root / "setup.cfg", root / "tox.ini") if path.is_file()
    ]
    python_configured = python_sources or bool(python_manifests) or "pytest" in pyproject_text.lower()
    if _python_runner(tokens):
        if not python_configured or not (python_tests or "pytest" in pyproject_text.lower() or any("pytest" in path.read_text(errors="ignore").lower() for path in python_manifests)):
            raise ValueError("Python test runner is not evidenced by this repository's source, tests, or configuration")
        return shlex.join(tokens)

    package_path = root / "package.json"
    if token0 in {"npm", "pnpm", "yarn"}:
        if not package_path.is_file():
            raise ValueError("JavaScript package manager command requires package.json")
        try:
            package = json.loads(package_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("package.json is invalid") from exc
        if not isinstance(package, dict):
            raise ValueError("package.json must contain an object")
        package_manager = package.get("packageManager", "")
        manager_name = package_manager.split("@", 1)[0] if isinstance(package_manager, str) else ""
        lockfiles = {"npm": "package-lock.json", "pnpm": "pnpm-lock.yaml", "yarn": "yarn.lock"}
        if manager_name and manager_name != token0:
            raise ValueError(f"repository declares {manager_name}, not {token0}")
        if not manager_name and (root / lockfiles[token0]).exists() is False:
            raise ValueError(f"no {token0} lockfile/package-manager declaration found")
        scripts = package.get("scripts", {})
        if not isinstance(scripts, dict) or not isinstance(scripts.get("test"), str):
            raise ValueError("package.json has no test script")
        if any(hook in scripts for hook in ("pretest", "posttest")):
            raise ValueError("pretest/posttest hooks require manual inspection and are not run")
        script_tokens = _lex_one_command(scripts["test"])
        if not _javascript_runner(script_tokens):
            raise ValueError("package test script is not a recognized test runner")
        valid_shape = (
            len(tokens) >= 2 and tokens[1] == "test"
        ) or (
            len(tokens) >= 3 and tokens[1:3] == ["run", "test"]
        )
        if not valid_shape:
            raise ValueError("use the repository-native package-manager test script")
        # Only explicitly forwarded runner args are accepted, and they must
        # preserve full-suite execution too.
        forwarded = []
        if len(tokens) > (2 if tokens[1] == "test" else 3):
            separator_index = 2 if tokens[1] == "test" else 3
            if tokens[separator_index] != "--":
                raise ValueError("runner arguments must follow the package-manager -- separator")
            forwarded = tokens[separator_index + 1:]
        if not _javascript_runner([*script_tokens, *forwarded]):
            raise ValueError("JavaScript test arguments must run the full configured suite")
        return shlex.join(tokens)

    if token0 in {"mvn", "gradle"}:
        if "--offline" not in tokens and "-o" not in tokens:
            raise ValueError("Java build/test commands must run offline")
        if token0 == "mvn" and not (root / "pom.xml").is_file():
            raise ValueError("Maven command requires pom.xml")
        if token0 == "gradle" and not any(
            (root / filename).is_file()
            for filename in ("build.gradle", "build.gradle.kts")
        ):
            raise ValueError("Gradle command requires build.gradle or build.gradle.kts")
        if token0 == "gradle":
            goals = []
            index = 1
            gradle_switches = {
                "--offline", "--no-daemon", "--console=plain", "-q", "--quiet",
                "--stacktrace", "--info", "--debug", "--continue", "--rerun-tasks",
                "--warning-mode=all", "--warning-mode=none",
            }
            while index < len(tokens):
                token = tokens[index]
                if token in {"--tests", "--exclude-task", "-x"}:
                    raise ValueError(f"Gradle option {token} skips or selects tests/tasks")
                if token in {"--project-dir", "-p", "--dry-run"} or token.startswith(("--project-dir=", "--init-script=", "-I")):
                    raise ValueError(f"Gradle option {token} can change or bypass the repository test run")
                if token.startswith(("--tests=", "--exclude-task=")):
                    raise ValueError(f"Gradle option {token} skips or selects tests/tasks")
                if token in {"--max-workers", "--console", "--warning-mode"}:
                    if index + 1 >= len(tokens):
                        raise ValueError(f"{token} requires an argument")
                    value = tokens[index + 1]
                    valid_value = (
                        (token == "--max-workers" and re.fullmatch(r"\d+", value))
                        or (token == "--console" and value in {"plain", "rich", "auto", "verbose"})
                        or (token == "--warning-mode" and value in {"all", "summary", "none", "fail"})
                    )
                    if not valid_value:
                        raise ValueError(f"unsupported value for Gradle option {token}")
                    index += 2
                    continue
                if re.fullmatch(r"--max-workers=\d+", token) or re.fullmatch(r"--console=(?:plain|rich|auto|verbose)", token) or re.fullmatch(r"--warning-mode=(?:all|summary|none|fail)", token):
                    index += 1
                    continue
                if token in gradle_switches:
                    index += 1
                    continue
                if token.startswith("-"):
                    raise ValueError(f"unsupported Gradle test option: {token}")
                goals.append(token)
                index += 1
        else:
            maven_tokens = tokens[1:]
            maven_switches = {
                "-o", "--offline", "-B", "--batch-mode", "-ntp",
                "--no-transfer-progress", "-q", "--quiet", "-V",
                "--show-version", "-e", "--errors", "-fae", "--fail-at-end",
                "-C", "--strict-checksums", "-c", "--lax-checksums",
            }
            goals = []
            index = 0
            while index < len(maven_tokens):
                token = maven_tokens[index]
                if token in {"-pl", "--projects", "-rf", "--resume-from", "-fn", "--fail-never", "-ff", "--fail-fast", "-N", "--non-recursive", "-P", "--activate-profiles", "-f", "--file"}:
                    raise ValueError(f"Maven option {token} can narrow, skip, or suppress tests")
                if token.startswith(("-pl", "-rf", "--projects=", "--resume-from=", "--activate-profiles=", "--file=")) or token in {"-ff"}:
                    raise ValueError(f"Maven option {token} can narrow, skip, or suppress tests")
                if token in {"-T", "--threads"}:
                    if index + 1 >= len(maven_tokens) or not re.fullmatch(r"[0-9]+(?:C)?", maven_tokens[index + 1]):
                        raise ValueError(f"invalid Maven thread count after {token}")
                    index += 2
                    continue
                if token.startswith("--threads="):
                    if not re.fullmatch(r"--threads=[0-9]+(?:C)?", token):
                        raise ValueError(f"invalid Maven thread count in {token}")
                    index += 1
                    continue
                if token.startswith("-T"):
                    if not re.fullmatch(r"-T[0-9]+(?:C)?", token):
                        raise ValueError(f"invalid Maven thread count in {token}")
                    index += 1
                    continue
                if token in maven_switches:
                    index += 1
                    continue
                if token.startswith("-D"):
                    benign_properties = {
                        "-dskiptests=false", "-dskipits=false", "-dmaven.test.skip=false",
                        "-dtestfailureignore=false", "-dmaven.test.failure.ignore=false",
                        "-dfailifnotests=true",
                    }
                    if (
                        not re.fullmatch(r"-Dstyle\.color=(?:never|always|auto)", token, re.I)
                        and token.lower() not in benign_properties
                        and not re.fullmatch(r"-D(?:revision|changelist|sha1)=[A-Za-z0-9._+-]+", token, re.I)
                    ):
                        raise ValueError(f"unrecognized Maven system property: {token}")
                    index += 1
                    continue
                if token.startswith("-"):
                    raise ValueError(f"unrecognized Maven test option: {token}")
                goals.append(token)
                index += 1
        accepted_goals = (["test"], ["clean", "test"], ["verify"], ["clean", "verify"])
        if goals not in accepted_goals:
            raise ValueError("Java command must request the complete test lifecycle")
        return shlex.join(tokens)

    raise ValueError("test command is not a supported, repository-verified runner")


def _run_verification_check(
    workspace, checks: list[dict], name: str, argv: list[str],
    plan: TestExecutionPlan,
) -> bool:
    """Run one check with the plan's interpreter and sanitized environment."""
    command = sandbox_test_command(plan, argv=argv)
    try:
        result = workspace.execute_command(command, timeout=600)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = _redact_sensitive_values(
            (stdout + ("\n" if stdout and stderr else "") + stderr)[-12000:]
        )
        code = result.exit_code
    except Exception as exc:
        code = None
        output = _redact_sensitive_values(
            f"Verification command raised {type(exc).__name__}: {exc}"
        )
    passed = code == 0
    checks.append({
        "name": name,
        "command": command,
        "exit_code": code,
        "output": output,
        "passed": passed,
    })
    return passed


def _python_verification_parts(
    tokens: list[str], interpreter: str | None = None
) -> tuple[list[str], str, list[str], str]:
    """Return wrapper, environment interpreter, runner argv, and test framework.

    Bare (non-wrapped) runners are rewritten to invoke the framework through
    the selected environment's own interpreter (``<interpreter> -m <framework>``)
    so the sanitized verifier environment never depends on the virtual
    environment's bin directory being on ``PATH``.

    Wrapped runners (uv, poetry, pipenv) select and activate their own
    environment, which may live outside the workspace. Their validated runner
    is therefore kept intact so an interpreter resolved elsewhere cannot
    bypass the wrapper's environment; the wrapper resolves its own ``python``.
    """
    wrapper: list[str] = []
    if tokens[0] == "uv":
        wrapper = tokens[:4]
        runner = tokens[4:]
        if runner and runner[0] == "--":
            runner = runner[1:]
    elif tokens[0] in {"poetry", "pipenv"}:
        wrapper = tokens[:2]
        runner = tokens[2:]
    else:
        runner = list(tokens)

    if wrapper:
        kind = _venv_runner_kind(runner[0]) if runner else None
        if runner[0] == "pytest" or kind == "pytest":
            env_interpreter = (
                _interpreter_sibling_python(runner[0]) if kind == "pytest" else "python"
            )
            return wrapper, env_interpreter, list(runner), "pytest"
        env_interpreter = runner[0]
        framework = runner[2]
        return wrapper, env_interpreter, list(runner), framework

    kind = _venv_runner_kind(runner[0]) if runner else None
    if runner[0] == "pytest" or kind == "pytest":
        if kind == "pytest":
            env_interpreter = _interpreter_sibling_python(runner[0])
            return (
                wrapper,
                env_interpreter,
                [env_interpreter, "-m", "pytest", *runner[1:]],
                "pytest",
            )
        if interpreter:
            return (
                wrapper,
                interpreter,
                [interpreter, "-m", "pytest", *runner[1:]],
                "pytest",
            )
        raise EnvironmentResolutionError(
            "A bare Python runner was validated without a resolved interpreter; "
            "refusing to use whatever pytest happens to be on PATH.",
            evidence={"tokens": tokens},
        )
    if (
        runner[0] in {"python", "python3"}
        or kind == "python"
        or (interpreter is not None and runner[0] == interpreter)
        or re.fullmatch(r"python3\.\d+", runner[0])
    ):
        if kind == "python":
            env_interpreter = runner[0]
        elif interpreter is not None and runner[0] in {"python", "python3"}:
            env_interpreter = interpreter
        else:
            env_interpreter = runner[0]
        framework = runner[2]
        return (
            wrapper,
            env_interpreter,
            [env_interpreter, "-m", framework, *runner[3:]],
            framework,
        )
    raise ValueError("Could not derive the Python runtime and test runner from the validated command")


# Interpreter introspection: report the environment associated with one
# interpreter, never environment values. `sys.prefix != sys.base_prefix`
# identifies a virtual environment even when its directory has an arbitrary
# name, so resolution does not depend on hardcoded paths.
_PYTHON_INTERPRETER_INTROSPECTION = (
    "import json, sys\n"
    "info = {\n"
    "    'executable': sys.executable,\n"
    "    'prefix': sys.prefix,\n"
    "    'base_prefix': sys.base_prefix,\n"
    "    'is_venv': sys.prefix != sys.base_prefix,\n"
    "    'version': list(sys.version_info[:3]),\n"
    "}\n"
    "try:\n"
    "    import pytest\n"
    "    info['has_pytest'] = True\n"
    "    info['pytest_version'] = getattr(pytest, '__version__', None)\n"
    "except Exception:\n"
    "    info['has_pytest'] = False\n"
    "print(json.dumps(info))\n"
)

_PYTHON_INTERPRETER_PROBE_SCRIPT = (
    "import json, os, shutil, subprocess, sys\n"
    "workspace = os.getcwd()\n"
    "config = CONFIG_JSON\n"
    "introspection = INTROSPECTION_JSON\n"
    "candidates = []\n"
    "seen = set()\n"
    "def add(raw, source):\n"
    "    if not raw:\n"
    "        return\n"
    "    raw = os.path.expanduser(str(raw))\n"
    "    if os.path.sep in raw:\n"
    "        path = os.path.abspath(raw)\n"
    "    else:\n"
    "        path = shutil.which(raw)\n"
    "        if not path:\n"
    "            return\n"
    "    # Deduplicate by path, not by realpath: a virtual environment's\n"
    "    # interpreter normally symlinks to its base interpreter, and collapsing\n"
    "    # them would hide the environment (and its packages) entirely.\n"
    "    key = os.path.normpath(path)\n"
    "    if key in seen:\n"
    "        return\n"
    "    seen.add(key)\n"
    "    candidates.append((path, source))\n"
    "for raw in config.get('candidates', []):\n"
    "    add(raw, 'command')\n"
    "for name in ('VIRTUAL_ENV', 'CONDA_PREFIX'):\n"
    "    root = os.environ.get(name)\n"
    "    if root:\n"
    "        add(os.path.join(root, 'bin', 'python'), 'active')\n"
    "for name in ('python3', 'python'):\n"
    "    path = shutil.which(name)\n"
    "    if path:\n"
    "        add(path, 'path')\n"
    "for name in ('.venv', 'venv', '.env', 'env'):\n"
    "    add(os.path.join(workspace, name, 'bin', 'python'), 'project')\n"
    "discovered = []\n"
    "skip = {'.git', 'node_modules', '__pycache__', '.pytest_cache'}\n"
    "for dirpath, dirnames, filenames in os.walk(workspace):\n"
    "    relative = os.path.relpath(dirpath, workspace)\n"
    "    depth = 0 if relative == '.' else relative.count(os.path.sep) + 1\n"
    "    dirnames[:] = [\n"
    "        name for name in dirnames\n"
    "        if name not in skip and not os.path.islink(os.path.join(dirpath, name))\n"
    "    ]\n"
    "    if depth >= 4:\n"
    "        dirnames[:] = []\n"
    "    if depth and 'pyvenv.cfg' in filenames:\n"
    "        discovered.append(dirpath)\n"
    "        dirnames[:] = []\n"
    "for root in sorted(discovered):\n"
    "    add(os.path.join(root, 'bin', 'python'), 'project')\n"
    "add(sys.executable, 'system')\n"
    "add('python3', 'system')\n"
    "add('python', 'system')\n"
    "report = {'python_candidates': [], 'probe_interpreter': sys.executable}\n"
    "for path, source in candidates:\n"
    "    if not (os.path.isfile(path) and os.access(path, os.X_OK)):\n"
    "        continue\n"
    "    info = {}\n"
    "    try:\n"
    "        probe = subprocess.run(\n"
    "            [path, '-c', introspection], capture_output=True, text=True, timeout=10,\n"
    "        )\n"
    "        output = (probe.stdout or '').strip()\n"
    "        if output:\n"
    "            info = json.loads(output.splitlines()[-1])\n"
    "    except Exception:\n"
    "        info = {}\n"
    "    if not info:\n"
    "        # Fall back to an import probe for interpreters that do not emit\n"
    "        # introspection JSON (for example minimal or shim interpreters).\n"
    "        try:\n"
    "            import_probe = subprocess.run(\n"
    "                [path, '-c', 'import pytest'], capture_output=True, timeout=10,\n"
    "            )\n"
    "            info = {'has_pytest': import_probe.returncode == 0}\n"
    "        except Exception:\n"
    "            info = {}\n"
    "    if not isinstance(info, dict):\n"
    "        info = {}\n"
    "    entry = {'path': path, 'source': source}\n"
    "    for key in ('executable', 'prefix', 'base_prefix', 'is_venv', 'has_pytest', 'pytest_version', 'version'):\n"
    "        if key in info:\n"
    "            entry[key] = info[key]\n"
    "    report['python_candidates'].append(entry)\n"
    "print(json.dumps(report))\n"
)


def _python_interpreter_probe_command(
    extra_candidates: list[str], executable: str = "python3"
) -> str:
    """Build a read-only sandbox probe that locates Python environments.

    The probe runs with a sanitized environment and reports interpreter paths
    and interpreter metadata only. It never prints environment values, so no
    secrets can leak through its output. ``executable`` is resolved from PATH
    by the sandbox shell, which lets callers fall back to ``python`` on images
    that do not provide ``python3``.
    """
    script = _PYTHON_INTERPRETER_PROBE_SCRIPT.replace(
        "CONFIG_JSON",
        json.dumps({"candidates": [str(item) for item in extra_candidates]}),
    ).replace(
        "INTROSPECTION_JSON",
        json.dumps(_PYTHON_INTERPRETER_INTROSPECTION),
    )
    return (
        "cd /workspace && env -i "
        'PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}" '
        'HOME="${HOME:-/tmp}" LANG="${LANG:-C.UTF-8}" '
        'VIRTUAL_ENV="${VIRTUAL_ENV:-}" CONDA_PREFIX="${CONDA_PREFIX:-}" '
        f"{executable} - <<'OPENHANDS_PYTHON_ENV_PROBE'\n"
        + script
        + "OPENHANDS_PYTHON_ENV_PROBE"
    )


def _valid_python_interpreter_path(value) -> bool:
    """Accept only absolute interpreter paths safe to embed in commands."""
    return (
        isinstance(value, str)
        and value.startswith("/")
        and ".." not in PurePosixPath(value).parts
        and re.search(r'["`$\\]', value) is None
        and not _has_control_characters(value)
    )


def _python_candidates_from_probe(output: str) -> list[dict]:
    """Normalize probe output into interpreter candidates.

    Accepts the structured ``python_candidates`` report and the legacy
    ``pytest_interpreter`` / ``environment_interpreter`` shape so older
    probes and test doubles keep working.
    """
    report = None
    for line in reversed((output or "").splitlines()):
        try:
            parsed = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            report = parsed
            break
    if report is None:
        return []

    candidates: list[dict] = []
    entries = report.get("python_candidates")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            if not _valid_python_interpreter_path(path):
                continue
            source = entry.get("source")
            if source not in {"command", "active", "project", "system", "path"}:
                source = "system"
            is_venv = entry.get("is_venv") is True
            if source == "active" and entry.get("is_venv") is False:
                # VIRTUAL_ENV/CONDA_PREFIX can be stale or point at a system
                # prefix. Only a genuine virtual environment is authoritative;
                # otherwise treat it as a system fallback so a real project
                # environment still wins.
                source = "system"
            candidates.append({
                "path": path,
                "source": source,
                "is_venv": is_venv,
                "has_pytest": entry.get("has_pytest") is True,
                "pytest_version": entry.get("pytest_version"),
                "version": entry.get("version"),
            })
        return candidates

    for key, has_pytest in (
        ("pytest_interpreter", True),
        ("environment_interpreter", False),
    ):
        path = report.get(key)
        if _valid_python_interpreter_path(path) and not any(
            candidate["path"] == path for candidate in candidates
        ):
            candidates.append({
                "path": path,
                "source": "legacy",
                "is_venv": False,
                "has_pytest": has_pytest,
            })
    return candidates


_PYTHON_CANDIDATE_SOURCE_ORDER = (
    "command", "active", "project", "path", "system", "legacy",
)


def _select_python_candidate(candidates: list[dict]) -> dict | None:
    """Select one environment without silently borrowing system Python.

    Priority is the environment named by the test command, an explicitly
    activated environment (``VIRTUAL_ENV``/``CONDA_PREFIX``), a project
    virtual environment (including ``pyvenv.cfg`` discoveries), a virtual
    environment merely present on ``PATH``, then system Python. Within a tier,
    an interpreter that can import pytest wins; otherwise the first candidate
    is kept, so verification reports the real environment's failure instead of
    switching to a different interpreter that happens to have pytest.
    """
    for source in _PYTHON_CANDIDATE_SOURCE_ORDER:
        group = [candidate for candidate in candidates if candidate["source"] == source]
        if not group:
            continue
        for candidate in group:
            if candidate["has_pytest"]:
                return candidate
        return group[0]
    return candidates[0] if candidates else None


def resolve_python_environment(
    workspace, tokens: list[str] | None = None
) -> dict | None:
    """Locate and describe the Python environment selected for the repository.

    This is the single source of truth for Python test execution. Evidence is
    gathered inside the sandbox in this order: an interpreter named by the
    validated test command; an explicitly activated environment
    (``VIRTUAL_ENV``/``CONDA_PREFIX``); a project virtual environment (common
    directory names plus any ``pyvenv.cfg`` marker under the workspace,
    including arbitrarily named or nested environments); a virtual environment
    merely present on ``PATH``; and finally a system interpreter. Selecting one
    environment and reusing its interpreter for every check and for the suite
    prevents the verifier from silently switching to system Python.

    Returns a dictionary containing ``interpreter``, ``venv_root``,
    ``has_pytest``, ``is_venv``, ``source``, and the inspected ``candidates``,
    or None when no interpreter could be identified.
    """
    extra_candidates: list[str] = []
    if tokens:
        kind = _venv_runner_kind(tokens[0])
        if kind == "python":
            extra_candidates.append(tokens[0])
        elif kind == "pytest":
            extra_candidates.append(_interpreter_sibling_python(tokens[0]))
    for executable in ("python3", "python"):
        try:
            result = workspace.execute_command(
                _python_interpreter_probe_command(extra_candidates, executable),
                timeout=120,
            )
        except Exception:
            continue
        candidates = _python_candidates_from_probe((result.stdout or "").strip())
        selected = _select_python_candidate(candidates)
        if selected is None:
            continue
        return {
            "interpreter": selected["path"],
            "venv_root": _venv_root_for_interpreter(selected["path"]),
            "has_pytest": selected["has_pytest"],
            "is_venv": selected["is_venv"],
            "source": selected["source"],
            "candidates": candidates,
        }
    return None


def resolve_python_interpreter(workspace, tokens: list[str] | None = None) -> str | None:
    """Return only the interpreter path from ``resolve_python_environment``."""
    environment = resolve_python_environment(workspace, tokens)
    return environment["interpreter"] if environment else None


def _split_python_run_shape(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Split a validated Python command into wrapper and runner tokens."""
    if tokens[0] == "uv":
        wrapper = tokens[:4]
        runner = tokens[4:]
        if runner and runner[0] == "--":
            runner = runner[1:]
        return wrapper, runner
    if tokens[0] in {"poetry", "pipenv"}:
        return tokens[:2], tokens[2:]
    return [], list(tokens)


def _absolute_workspace_path(path: str) -> str:
    """Make a sandbox-relative path absolute without leaving the workspace."""
    return path if path.startswith("/") else "/workspace/" + path


def _build_sandbox_env(venv_root: str | None = None) -> dict[str, str]:
    """Build the deterministic, non-secret environment for a plan.

    Host secrets are never copied. Only a small allowlist of display/CI
    controls may be inherited, and only when the value is safe to embed.
    """
    default_path = ":".join([*_SANDBOX_USER_BIN, SANDBOX_PATH_DEFAULT])
    env = dict(SANDBOX_ENV_DEFAULTS)
    if venv_root:
        env["PATH"] = f"{venv_root}/bin:{default_path}"
        env["VIRTUAL_ENV"] = venv_root
    else:
        env["PATH"] = default_path
    for name in _SANDBOX_OPTIONAL_ENV:
        value = os.environ.get(name)
        if (
            value
            and not _UNSAFE_ENV_VALUE_RE.search(value)
            and not _has_control_characters(value)
        ):
            env[name] = value
    return env


def _python_execution_plan(
    tokens: list[str],
    interpreter: str | None = None,
    requested_command: str | None = None,
    environment: dict | None = None,
) -> TestExecutionPlan:
    """Create a Python plan from validated tokens and optional probe evidence."""
    wrapper, runner = _split_python_run_shape(tokens)
    requested = requested_command or shlex.join(tokens)

    if wrapper:
        _wrapper, _interpreter, resolved_runner, framework = _python_verification_parts(
            tokens, None
        )
        return TestExecutionPlan(
            # Wrappers select their own environment; preserve the validated
            # argv exactly (including any ``--`` separator) for the suite.
            argv=tuple(tokens),
            env=_build_sandbox_env(None),
            interpreter=None,
            venv_root=None,
            framework=framework,
            ecosystem="python",
            source="wrapper",
            evidence={
                "requested_command": requested,
                "resolution": "wrapper selects its own environment",
                "wrapper": wrapper,
            },
            wrapper=tuple(wrapper),
            runner=tuple(resolved_runner),
        )

    # A command that names its own virtual environment is authoritative: it
    # must run in that environment even if the probe selected another one.
    # Normalize relative workspace runners to absolute paths so the plan does
    # not depend on the shell's working directory.
    kind = _venv_runner_kind(runner[0]) if runner else None
    normalized = list(tokens)
    command_interpreter = None
    if kind == "pytest":
        command_interpreter = _absolute_workspace_path(
            _interpreter_sibling_python(runner[0])
        )
        normalized[0] = _absolute_workspace_path(normalized[0])
    elif kind == "python":
        command_interpreter = _absolute_workspace_path(runner[0])
        normalized[0] = command_interpreter

    evidence = {"requested_command": requested}
    if command_interpreter:
        selected = command_interpreter
        venv_root = _venv_root_for_interpreter(selected)
        source = (environment or {}).get("source") or "command"
        if environment is not None and environment.get("interpreter") != selected:
            evidence["command_environment_override"] = {
                "probe_interpreter": environment.get("interpreter"),
                "command_interpreter": selected,
            }
    else:
        selected = (environment or {}).get("interpreter")
        if not selected and interpreter and _valid_python_interpreter_path(interpreter):
            selected = interpreter
        venv_root = (environment or {}).get("venv_root") or _venv_root_for_interpreter(
            selected
        )
        source = (environment or {}).get("source") or "command"

    if not selected or not _valid_python_interpreter_path(selected):
        raise EnvironmentResolutionError(
            "The validated Python command does not name a resolvable interpreter "
            "and no sandbox interpreter was resolved.",
            evidence={
                "requested_command": requested,
                "interpreter": interpreter,
                "python_environment": environment,
            },
        )

    wrapper, resolved_interpreter, resolved_runner, framework = (
        _python_verification_parts(normalized, selected)
    )
    if venv_root is None:
        venv_root = _venv_root_for_interpreter(resolved_interpreter)
    evidence["resolved_interpreter"] = resolved_interpreter
    if environment is not None:
        evidence["python_environment"] = environment
        if (
            framework == "pytest"
            and environment.get("interpreter") == resolved_interpreter
            and environment.get("has_pytest") is False
        ):
            evidence["framework_available"] = False
    return TestExecutionPlan(
        argv=tuple([*wrapper, *resolved_runner]),
        env=_build_sandbox_env(venv_root),
        interpreter=resolved_interpreter,
        venv_root=venv_root,
        framework=framework,
        ecosystem="python",
        source=source,
        evidence=evidence,
        wrapper=tuple(wrapper),
        runner=tuple(resolved_runner),
    )


def test_execution_plan_from_command(
    command: str | TestExecutionPlan,
    python_interpreter: str | None = None,
) -> TestExecutionPlan:
    """Build a plan without probing a workspace.

    Used for non-Python commands, reporting, and as the strict compatibility
    layer for ``sandbox_test_command``. A Python command that does not name a
    resolvable interpreter raises ``EnvironmentResolutionError`` instead of
    silently falling back to a PATH lookup.
    """
    if isinstance(command, TestExecutionPlan):
        return command
    tokens = shlex.split(command) if isinstance(command, str) else [str(item) for item in command]
    if not tokens:
        raise EnvironmentResolutionError("The test command is empty.")
    if _python_runner(tokens):
        return _python_execution_plan(
            tokens, interpreter=python_interpreter, requested_command=shlex.join(tokens)
        )
    if tokens[0] in {"npm", "pnpm", "yarn"}:
        ecosystem, framework = "javascript", Path(tokens[0]).name
    elif tokens[0] in {"mvn", "gradle"}:
        ecosystem, framework = "java", tokens[0]
    else:
        raise EnvironmentResolutionError(
            "The command is not a supported test runner and cannot be rendered "
            "without an execution plan.",
            evidence={"requested_command": shlex.join(tokens)},
        )
    return TestExecutionPlan(
        argv=tuple(tokens),
        env=_build_sandbox_env(None),
        interpreter=None,
        venv_root=None,
        framework=framework,
        ecosystem=ecosystem,
        source="command",
        evidence={"requested_command": shlex.join(tokens)},
    )


# This helper is a constructor, not a test; prevent pytest collection when a
# test module imports it directly.
test_execution_plan_from_command.__test__ = False


def resolve_test_execution_plan(
    workspace, repository_root: Path, test_command: str
) -> TestExecutionPlan:
    """Resolve the one execution plan used for verification and execution.

    Raises ``EnvironmentResolutionError`` with structured evidence when no safe
    plan exists, so callers fail closed instead of executing a bare command
    against an unknown interpreter.
    """
    validated = validate_test_command(test_command, repository_root)
    tokens = shlex.split(validated)

    if _python_runner(tokens):
        wrapper, _runner = _split_python_run_shape(tokens)
        if wrapper:
            return _python_execution_plan(
                tokens, requested_command=validated
            )
        try:
            environment = resolve_python_environment(workspace, tokens)
        except Exception as exc:
            raise EnvironmentResolutionError(
                "Python interpreter probing could not complete.",
                evidence={
                    "requested_command": validated,
                    "probe_error": f"{type(exc).__name__}: {exc}",
                },
            ) from exc
        if environment is None:
            raise EnvironmentResolutionError(
                "No Python interpreter could be resolved inside the sandbox.",
                evidence={
                    "requested_command": validated,
                    "tokens": tokens,
                    "probe": "no usable interpreter returned",
                },
            )
        return _python_execution_plan(
            tokens,
            interpreter=environment["interpreter"],
            requested_command=validated,
            environment=environment,
        )

    if tokens[0] in {"npm", "pnpm", "yarn"}:
        _package, script_tokens = _javascript_script_tokens(repository_root)
        ecosystem, framework = "javascript", _javascript_framework(script_tokens)
    elif tokens[0] in {"mvn", "gradle"}:
        ecosystem, framework = "java", tokens[0]
    else:
        raise EnvironmentResolutionError(
            "The validated command could not be mapped to a supported ecosystem.",
            evidence={"requested_command": validated, "tokens": tokens},
        )
    return TestExecutionPlan(
        argv=tuple(tokens),
        env=_build_sandbox_env(None),
        interpreter=None,
        venv_root=None,
        framework=framework,
        ecosystem=ecosystem,
        source="command",
        evidence={"requested_command": validated},
    )


def _javascript_script_tokens(repository_root: Path) -> tuple[dict, list[str]]:
    package = json.loads((repository_root / "package.json").read_text(encoding="utf-8"))
    return package, _lex_one_command(package["scripts"]["test"])


def _javascript_framework(tokens: list[str]) -> str:
    while tokens and Path(tokens[0]).name == "cross-env":
        index = 1
        while index < len(tokens) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[index]):
            index += 1
        tokens = tokens[index:]
    name = Path(tokens[0]).name
    if name == "react-scripts":
        return "jest"
    if name == "vue-cli-service":
        return "mocha"
    if name == "ng":
        return "angular"
    return name


def _javascript_collection_runner(tokens: list[str]) -> list[str] | None:
    """Return a safe, non-executing collector argv for supported JS runners."""
    tokens = list(tokens)
    while tokens and Path(tokens[0]).name == "cross-env":
        index = 1
        while index < len(tokens) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[index]):
            index += 1
        tokens = tokens[index:]
    name = Path(tokens[0]).name
    if name in {"jest", "react-scripts"}:
        return [*tokens, "--listTests"]
    if name == "mocha":
        return [*tokens, "--dry-run"]
    if name == "vue-cli-service":
        return [*tokens, "--dry-run"]
    if name == "vitest":
        # `vitest list` is separate from `vitest run`; do not accidentally
        # pass "list" as a file filter after the run subcommand.
        return [tokens[0], "list"]
    return None


def _unresolved_environment_verification(
    test_command: str, error: EnvironmentResolutionError
) -> dict:
    """Fail-safe verification result when no execution plan can be built."""
    evidence = error.evidence
    message = f"environment_unresolved: {error}"
    return {
        "ecosystem": "python" if evidence.get("tokens") else "unknown",
        "repository_manifests": [],
        "selected_test_command": test_command,
        "python_interpreter": None,
        "python_environment": evidence.get("python_environment"),
        "execution_plan": None,
        "plan_digest": None,
        "environment_unresolved": evidence,
        "ready": False,
        "checks": [],
        "failures": [message],
        "blockers": [message],
        "limitations": [],
    }


def verify_test_environment(
    workspace,
    repository_root: Path,
    test_command: str,
    plan: TestExecutionPlan | dict | None = None,
) -> dict:
    """Independently check runtime, runner, dependency health and test discovery.

    Every check uses the same ``TestExecutionPlan`` that will run the suite.
    When no plan is supplied one is resolved once here; if resolution fails the
    result is a fail-safe ``environment_unresolved`` report and no command is
    executed. Collection/listing is recorded separately and is never
    sufficient by itself to set ``ready``. Test functions are not executed by
    these checks.
    """
    checks: list[dict] = []
    failures: list[str] = []
    limitations: list[str] = []
    ecosystem = "unknown"
    python_interpreter: str | None = None
    python_environment: dict | None = None
    validated = validate_test_command(test_command, repository_root)
    tokens = shlex.split(validated)

    if isinstance(plan, dict):
        plan = TestExecutionPlan.from_dict(plan)
    if plan is None:
        try:
            plan = resolve_test_execution_plan(workspace, repository_root, validated)
        except EnvironmentResolutionError as exc:
            return _unresolved_environment_verification(validated, exc)

    ecosystem = plan.ecosystem
    python_interpreter = plan.interpreter
    python_environment = (plan.evidence or {}).get("python_environment")

    if _python_runner(tokens) and plan.source != "wrapper" and not python_interpreter:
        return _unresolved_environment_verification(
            validated,
            EnvironmentResolutionError(
                "The execution plan does not contain a Python interpreter for "
                "a non-wrapped runner.",
                evidence={"tokens": tokens, "execution_plan": plan.to_dict()},
            ),
        )

    def check(name: str, argv: list[str]) -> bool:
        passed = _run_verification_check(
            workspace, checks, name, argv, plan
        )
        if not passed:
            entry = checks[-1]
            failures.append(
                f"{name}: command {entry['command']!r} exited with "
                f"{entry['exit_code']!r}; output: {entry['output'] or '(no output)'}"
            )
        return passed

    def check_collection(name: str, argv: list[str]) -> bool:
        passed = check(name, argv)
        if not passed:
            return False
        output = checks[-1]["output"]
        no_tests = re.search(
            r"(?:\b0\s+(?:tests?|items?)\b|no tests? (?:found|collected|were found))",
            output,
            re.I,
        )
        if not output.strip() or no_tests:
            checks[-1]["passed"] = False
            checks[-1]["collection_evidence_error"] = "Collector returned no evidence of discovered tests."
            failures.append(
                f"{name}: collector returned no evidence of discovered tests; "
                f"output: {output or '(no output)'}"
            )
            return False
        return True

    if _python_runner(tokens):
        ecosystem = "python"
        wrapper, interpreter, runner, framework = _python_verification_parts(
            tokens, python_interpreter
        )
        framework_missing = (plan.evidence or {}).get("framework_available") is False
        if framework_missing:
            failures.append(
                f"python_test_runner: selected interpreter {python_interpreter!r} "
                f"cannot import {framework}; refusing to borrow another "
                "environment's test framework."
            )
        runtime_ok = check("python_runtime", [*wrapper, interpreter, "--version"])
        runner_version_args = ["-h"] if framework == "unittest" else ["--version"]
        runner_ok = check("python_test_runner", [*wrapper, *runner, *runner_version_args])
        pip_probe = check(
            "python_pip_availability",
            [*wrapper, interpreter, "-c", "import importlib.util; print('available' if importlib.util.find_spec('pip') else 'unavailable')"],
        )
        pip_check = False
        pip_available = pip_probe and checks[-1]["output"].strip() == "available"
        if pip_available:
            pip_check = check("python_dependency_consistency", [*wrapper, interpreter, "-m", "pip", "check"])
        else:
            limitations.append("pip is unavailable in the selected Python environment; dependency consistency relies on imports during collection.")
        if framework == "pytest":
            collection_ok = check_collection("python_test_collection", [*wrapper, *runner, "--collect-only", "-q"])
        else:
            discovery = (
                "import unittest; suite=unittest.defaultTestLoader.discover('.'); "
                "count=suite.countTestCases(); print(f'discovered {count} tests'); "
                "raise SystemExit(0 if count else 5)"
            )
            collection_ok = check_collection("python_test_collection", [*wrapper, interpreter, "-c", discovery])
        required_ok = (
            runtime_ok
            and runner_ok
            and pip_probe
            and (pip_check or not pip_available)
            and collection_ok
            and not framework_missing
        )
    elif tokens[0] in {"npm", "pnpm", "yarn"}:
        ecosystem = "javascript"
        manager = tokens[0]
        package, script_tokens = _javascript_script_tokens(repository_root)
        framework = _javascript_framework(script_tokens)
        node_ok = check("javascript_runtime", ["node", "--version"])
        manager_env = ["env", "COREPACK_ENABLE_NETWORK=0"] if manager in {"pnpm", "yarn"} else []
        manager_ok = check("javascript_package_manager", [*manager_env, manager, "--version"])
        runner_probe = [manager, "test"]
        if manager == "yarn":
            runner_probe.append("--version")
        else:
            runner_probe.extend(["--", "--version"])
        runner_ok = check("javascript_test_runner", [*manager_env, *runner_probe])
        if manager == "npm":
            dependency_ok = check("javascript_dependency_consistency", ["npm", "ls", "--depth=0", "--offline"])
        elif manager == "pnpm":
            dependency_ok = check("javascript_dependency_consistency", [*manager_env, "pnpm", "list", "--depth=0", "--offline"])
        else:
            lockfile = repository_root / "yarn.lock"
            lock_text = lockfile.read_text(encoding="utf-8", errors="ignore") if lockfile.is_file() else ""
            if "__metadata:" in lock_text:
                limitations.append("Yarn Berry does not provide a read-only dependency-integrity check; runner resolution and collection are verified.")
                dependency_ok = True
            else:
                dependency_ok = check("javascript_dependency_consistency", [*manager_env, "yarn", "check", "--integrity"])
        collection_runner = _javascript_collection_runner(script_tokens)
        if collection_runner is None:
            collection_ok = True
            limitations.append(f"Independent test collection is unavailable for JavaScript runner {framework}; runtime, package and runner checks were performed without executing tests.")
        else:
            if framework == "vitest":
                if manager == "npm":
                    collection_command = ["npm", "exec", "--offline", "--no", "--", *collection_runner]
                elif manager == "pnpm":
                    collection_command = [*manager_env, "pnpm", "exec", *collection_runner]
                else:
                    collection_command = [*manager_env, "yarn", "run", *collection_runner]
            else:
                if manager == "yarn":
                    collection_command = [*manager_env, "yarn", "test", collection_runner[-1]]
                else:
                    collector_option = collection_runner[-1]
                    collection_command = [*manager_env, manager, "test", "--", collector_option]
            collection_ok = check_collection("javascript_test_collection", collection_command)
        required_ok = node_ok and manager_ok and runner_ok and dependency_ok and collection_ok
    elif tokens[0] in {"mvn", "gradle"}:
        ecosystem = "java"
        manager = tokens[0]
        java_ok = check("java_runtime", ["java", "-version"])
        manager_ok = check("java_build_tool", [manager, "--version"])
        if manager == "mvn":
            build_ok = check("java_dependencies_and_test_compile", ["mvn", "--offline", "--batch-mode", "test-compile"])
        else:
            build_ok = check("java_dependencies_and_test_compile", ["gradle", "--offline", "testClasses"])
        required_ok = java_ok and manager_ok and build_ok
        limitations.append("The Java build tool has no supported non-executing test collection check; test sources were compiled, but framework discovery will be confirmed by the actual suite run.")
        checks.append({
            "name": "java_test_collection",
            "command": None,
            "exit_code": None,
            "output": "Not independently available without executing test methods.",
            "passed": None,
        })
    else:
        required_ok = False
        failures.append("The selected test command could not be associated with a supported repository ecosystem.")

    manifests = {
        "python": [path.name for path in repository_root.glob("requirements*.txt")]
        + [name for name in ("pyproject.toml", "setup.cfg", "tox.ini") if (repository_root / name).is_file()],
        "javascript": [name for name in ("package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock") if (repository_root / name).is_file()],
        "java": [name for name in ("pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts") if (repository_root / name).is_file()],
    }.get(ecosystem, [])

    return {
        "ecosystem": ecosystem,
        "repository_manifests": manifests,
        "selected_test_command": validated,
        "python_interpreter": python_interpreter,
        "python_environment": python_environment,
        "execution_plan": plan.to_dict(),
        "plan_digest": plan.digest,
        "plan_source": plan.source,
        "ready": required_ok,
        "checks": checks,
        "failures": failures,
        "blockers": failures.copy(),
        "limitations": limitations,
    }


def _run_agent_turn(
    conversation, run_timeout: float | None = None, info: dict | None = None
) -> dict:
    """Run one agent turn with a supported API timeout when available."""
    info = info if isinstance(info, dict) else {}
    info.update({
        "timeout_enforcement": "not_requested",
        "timeout_seconds": None,
        "timeout_supported": False,
    })
    if run_timeout is None:
        conversation.run()
        return info

    runner = getattr(conversation, "run", None)
    try:
        accepts_timeout = "timeout" in inspect.signature(runner).parameters
    except (TypeError, ValueError):
        accepts_timeout = False

    if accepts_timeout:
        info.update({
            "timeout_enforcement": "api_timeout",
            "timeout_seconds": float(run_timeout),
            "timeout_supported": True,
        })
        runner(timeout=float(run_timeout))
    else:
        # RemoteConversation (used by DockerWorkspace) supports timeout. Other
        # conversation implementations are bounded only by the caller's hard
        # deadline and must not start a second turn.
        info.update({
            "timeout_enforcement": "unsupported_by_conversation",
            "timeout_seconds": None,
            "timeout_supported": False,
        })
        runner()
    return info


def _agent_turn_error_is_timeout(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "timeout" in name or "timed out" in text or "timeout" in text


def _interrupt_agent_turn(conversation) -> bool:
    """Request cancellation of an active turn when the API supports it."""
    interrupt = getattr(conversation, "interrupt", None)
    if interrupt is None:
        return False
    try:
        interrupt()
        return True
    except Exception:
        return False


def run_environment_recovery(
    workspace,
    conversation,
    verification: dict,
    verify_again,
    max_attempts: int = 2,
    integrity_guard=None,
    sleep_fn=time.sleep,
    run_timeout: float | None = None,
) -> tuple[dict, list[dict]]:
    """Allow bounded, evidence-led repairs for environment preparation errors.

    ``verify_again`` must independently validate runtimes, dependencies and
    test discovery. Repairs never authorize editing test files, and integrity
    is checked both before and after each agent turn.
    """
    attempts: list[dict] = []
    plan_changes: list[dict] = []
    current_plan_digest = verification.get("plan_digest")
    limit = max(0, min(int(max_attempts), 2))
    terminal_markers = re.compile(
        r"(?:credential|authentication|unauthorized|permission denied|"
        r"docker (?:is not available|daemon)|connection refused|service unavailable)",
        re.I,
    )
    for attempt in range(1, limit + 1):
        failures = verification.get("failures", [])
        blockers = verification.get("blockers", [])
        evidence = _redact_sensitive_values(
            "\n".join(str(item) for item in [*failures, *blockers])
        )
        if terminal_markers.search(evidence):
            attempts.append({"attempt": attempt, "result": "blocked", "reason": evidence})
            break
        if integrity_guard is not None:
            before = integrity_guard()
            if not before.get("unchanged"):
                attempts.append({"attempt": attempt, "result": "blocked", "reason": "test integrity guard failed before recovery", "integrity": before})
                break
        prompt = f"""
Environment preparation failed independent verification. Diagnose the captured
runtime, dependency, module-resolution, build/configuration, and collection
output; apply a minimal evidence-backed fix inside this disposable sandbox.
You may install a dependency only after identifying it from repository
manifests/lockfiles and observed errors, using that repository's package
manager (pip/uv for Python; npm/pnpm/yarn for JavaScript; Maven/Gradle for
Java). Preserve pinned versions and lockfiles, use the declared manager, and
do not guess package names or versions. Do not modify, delete, skip,
rename, chmod, or weaken any existing or generated test. Do not change the
harness, workflow, security controls, or protected files. Do not retry for
missing credentials, external service outages, Docker/permission boundaries,
or ambiguous packages. Do not claim readiness; independent verification follows.

Verification failures:
{evidence[-24000:]}
"""
        turn_error: Exception | None = None
        turn_info = {
            "timeout_enforcement": "not_started",
            "timeout_seconds": run_timeout,
            "timeout_supported": False,
        }
        interrupted = False
        try:
            conversation.send_message(prompt)
            try:
                _run_agent_turn(conversation, run_timeout, turn_info)
            except Exception as exc:
                turn_error = exc
                if _agent_turn_error_is_timeout(exc):
                    interrupted = _interrupt_agent_turn(conversation)

            # Cancellation or timeout never skips artifact-integrity checks.
            integrity = integrity_guard() if integrity_guard is not None else {"unchanged": True}
            if not integrity.get("unchanged"):
                attempts.append({
                    "attempt": attempt,
                    "result": "blocked",
                    "reason": "test integrity guard failed after recovery",
                    "integrity": integrity,
                    "turn": turn_info,
                })
                break
            if turn_error is not None:
                attempts.append({
                    "attempt": attempt,
                    "result": "timeout" if _agent_turn_error_is_timeout(turn_error) else "error",
                    "error": _redact_sensitive_values(
                        f"{type(turn_error).__name__}: {turn_error}"
                    ),
                    "integrity": integrity,
                    "turn": turn_info,
                    "interrupted": interrupted,
                })
                break
            verification = verify_again()
            record = {
                "attempt": attempt,
                "result": "ready" if verification.get("ready") else "still_blocked",
                "failures": verification.get("failures", []),
                "integrity": integrity,
                "turn": turn_info,
            }
            new_plan_digest = verification.get("plan_digest")
            if current_plan_digest or new_plan_digest:
                changed = current_plan_digest != new_plan_digest
                record.update({
                    "plan_changed": changed,
                    "previous_plan_digest": current_plan_digest,
                    "plan_digest": new_plan_digest,
                    "plan_source": verification.get("plan_source"),
                })
                if changed:
                    record["execution_plan"] = verification.get("execution_plan")
                    plan_changes.append({
                        "attempt": attempt,
                        "previous_plan_digest": current_plan_digest,
                        "plan_digest": new_plan_digest,
                        "plan_source": verification.get("plan_source"),
                    })
                current_plan_digest = new_plan_digest
            attempts.append(record)
            if verification.get("ready"):
                break
            if attempt < limit:
                sleep_fn(min(0.5 * (2 ** (attempt - 1)), 2.0))
        except Exception as exc:
            attempts.append({
                "attempt": attempt,
                "result": "error",
                "error": _redact_sensitive_values(f"{type(exc).__name__}: {exc}"),
            })
            break
    verification["recovery_attempts"] = attempts
    verification["plan_changes"] = plan_changes
    return verification, attempts


def _redact_sensitive_values(text: str) -> str:
    """Remove secret-bearing environment values before prompts/artifacts."""
    for name, value in os.environ.items():
        if re.search(r"(?:TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|AUTH)", name, re.I) and value:
            text = text.replace(value, "[REDACTED]")
    return text


_SYSTEM_INTERPRETER_ROOTS = {
    "/", "/usr", "/usr/local", "/bin", "/sbin", "/lib", "/lib64", "/etc",
}


def _has_control_characters(value: str) -> bool:
    """Reject newlines and other control characters from embedded paths."""
    return any(ord(char) < 0x20 or ord(char) == 0x7F for char in value)


def _venv_root_for_interpreter(interpreter: str | None) -> str | None:
    """Derive a virtual-environment root from an interpreter path, safely.

    Returns None for bare/system interpreter names, system interpreter roots,
    and paths containing characters that could break out of the generated
    environment assignment.
    """
    if not interpreter:
        return None
    parts = PurePosixPath(interpreter).parts
    if (
        len(parts) < 3
        or ".." in parts
        or parts[-2] != "bin"
        or not re.fullmatch(r"python(?:3(?:\.\d+)?)?", parts[-1])
    ):
        return None
    root = str(PurePosixPath(*parts[:-2]))
    if not root or re.search(r'["`$\\]', root) or _has_control_characters(root):
        return None
    if not root.startswith("/"):
        root = "/workspace/" + root
    if root in _SYSTEM_INTERPRETER_ROOTS:
        return None
    return root


def _render_plan_environment(plan: TestExecutionPlan) -> list[str]:
    """Render ``env -i`` assignments from a plan without ever leaking secrets."""
    environment_args = ["env", "-i"]
    path_value = plan.env.get(
        "PATH", ":".join([*_SANDBOX_USER_BIN, SANDBOX_PATH_DEFAULT])
    )
    if _UNSAFE_ENV_VALUE_RE.search(path_value) or _has_control_characters(path_value):
        raise ValueError("execution plan contains an unsafe PATH value")

    if plan.venv_root:
        if _UNSAFE_ENV_VALUE_RE.search(plan.venv_root) or _has_control_characters(
            plan.venv_root
        ):
            raise ValueError("execution plan contains an unsafe virtual environment root")
        prefix = plan.venv_root + "/bin:"
        inherited = path_value[len(prefix):] if path_value.startswith(prefix) else path_value
        environment_args.append(f'PATH="{plan.venv_root}/bin:${{PATH:-{inherited}}}"')
    else:
        environment_args.append(f'PATH="${{PATH:-{path_value}}}"')

    for name in sorted(plan.env):
        if name == "PATH":
            continue
        value = plan.env[name]
        if (
            not _ENV_NAME_RE.fullmatch(name)
            or _UNSAFE_ENV_VALUE_RE.search(value)
            or _has_control_characters(value)
        ):
            raise ValueError(f"execution plan contains an unsafe value for {name}")
        environment_args.append(f'{name}="{value}"')
    return environment_args


def sandbox_test_command(
    plan: TestExecutionPlan | str,
    python_interpreter: str | None = None,
    argv: list[str] | None = None,
) -> str:
    """Render one sandbox command from a mandatory execution plan.

    ``plan`` is the single source of truth for the interpreter, virtual
    environment, and sanitized environment. A raw string is accepted for
    backwards compatibility only when it resolves statically (an explicit
    interpreter or a workspace venv path); a bare Python runner without a
    resolved interpreter raises ``EnvironmentResolutionError`` instead of
    being executed against an unknown environment.

    ``argv`` overrides only the command arguments (for verification probes);
    the plan's environment is always used. No secret values are forwarded.
    """
    if isinstance(plan, TestExecutionPlan):
        execution_plan = plan
    else:
        execution_plan = test_execution_plan_from_command(plan, python_interpreter)

    command_argv = list(argv) if argv is not None else list(execution_plan.argv)
    if not command_argv:
        raise EnvironmentResolutionError("The test execution plan has no argv.")

    environment_args = _render_plan_environment(execution_plan)
    return (
        "cd /workspace && "
        + " ".join(environment_args)
        + " "
        + shlex.join(command_argv)
    )


def generated_tests_manifest_from_archive(archive_path: Path) -> dict:
    """Validate and fingerprint a generated-tests tar without extracting it."""
    if archive_path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("Generated-test snapshot archive exceeds 100 MiB")
    manifest = {}
    root_found = False
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 1000:
            raise ValueError("Generated-test snapshot contains too many entries")
        total_size = 0
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("tests", "generated"):
                raise ValueError(f"Unsafe generated-test archive path: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported generated-test archive entry: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in generated tests: {member.name}")
            if member.mode & 0o7000:
                raise ValueError(f"Special permissions are not allowed in generated tests: {member.name}")
            total_size += member.size
            if total_size > 100 * 1024 * 1024:
                raise ValueError("Generated-test snapshot exceeds 100 MiB uncompressed")
            if path.parts == ("tests", "generated"):
                root_found = member.isdir()
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            key = path.as_posix().rstrip("/")
            if key in manifest:
                raise ValueError(f"Duplicate generated-test archive entry: {member.name}")
            value = {"type": "directory" if member.isdir() else "file", "mode": member.mode & 0o777}
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Could not read generated test: {member.name}")
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                value["sha256"] = digest.hexdigest()
            manifest[key] = value
    if not root_found:
        raise ValueError("Generated-test archive has no tests/generated directory")
    return manifest


def capture_generated_tests_snapshot(workspace, archive_path: Path) -> dict:
    """Save generated tests outside the agent workspace as the trusted baseline."""
    archive_in_workspace = "/tmp/openhands-generated-tests-snapshot.tar.gz"
    captured = workspace.execute_command(
        f"rm -f {archive_in_workspace} && "
        f"tar -czf {archive_in_workspace} -C /workspace tests/generated",
        timeout=120,
    )
    if captured.exit_code != 0:
        raise RuntimeError(f"Could not snapshot generated tests: {captured.stderr or captured.stdout}")
    download = workspace.file_download(
        source_path=archive_in_workspace,
        destination_path=str(archive_path),
    )
    if not download.success:
        raise RuntimeError(f"Could not download generated-test snapshot: {download.error}")
    return generated_tests_manifest_from_archive(archive_path)


_REPOSITORY_TEST_CAPTURE_COMMAND = (
    "find /workspace "
    "-type d \\( -name .git -o -name node_modules -o -name .venv "
    "-o -name __pycache__ -o -name target -o -name build -o -name dist "
    "-o -path /workspace/tests/generated \\) -prune -o "
    "-type f \\( -path '*/tests/*' -o -path '*/test/*' "
    "-o -path '*/__tests__/*' -o -path '*/src/test/*' "
    "-o -name 'test_*.py' -o -name '*_test.py' "
    "-o -name '*.test.*' -o -name '*.spec.*' \\) -printf '%P\\0' | "
    "tar -czf /tmp/openhands-repository-tests.tar.gz "
    "-C /workspace --null -T -"
)
_REPOSITORY_TEST_DELETE_COMMAND = (
    "find /workspace "
    "-type d \\( -name .git -o -name node_modules -o -name .venv "
    "-o -name __pycache__ -o -name target -o -name build -o -name dist "
    "-o -path /workspace/tests/generated \\) -prune -o "
    "-type f \\( -path '*/tests/*' -o -path '*/test/*' "
    "-o -path '*/__tests__/*' -o -path '*/src/test/*' "
    "-o -name 'test_*.py' -o -name '*_test.py' "
    "-o -name '*.test.*' -o -name '*.spec.*' \\) -delete"
)


def repository_test_manifest_from_archive(archive_path: Path) -> dict:
    """Fingerprint conventional existing repository tests without extracting them."""
    if archive_path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("Repository-test snapshot archive exceeds 100 MiB")
    manifest = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 5000:
            raise ValueError("Repository-test snapshot contains too many entries")
        total_size = 0
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or not path.parts or ".." in path.parts:
                raise ValueError(f"Unsafe repository-test archive path: {member.name}")
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"Unsupported repository-test archive entry: {member.name}")
            total_size += member.size
            if total_size > 100 * 1024 * 1024:
                raise ValueError("Repository-test snapshot exceeds 100 MiB uncompressed")
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            key = path.as_posix()
            if key in manifest:
                raise ValueError(f"Duplicate repository-test archive entry: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Could not read repository test: {member.name}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            manifest[key] = {"mode": member.mode & 0o777, "sha256": digest.hexdigest()}
    return manifest


def capture_repository_tests_snapshot(workspace, archive_path: Path) -> dict:
    captured = workspace.execute_command(_REPOSITORY_TEST_CAPTURE_COMMAND, timeout=120)
    if captured.exit_code != 0:
        raise RuntimeError(f"Could not snapshot repository tests: {captured.stderr or captured.stdout}")
    download = workspace.file_download(
        source_path="/tmp/openhands-repository-tests.tar.gz",
        destination_path=str(archive_path),
    )
    if not download.success:
        raise RuntimeError(f"Could not download repository-test snapshot: {download.error}")
    return repository_test_manifest_from_archive(archive_path)


def ensure_repository_tests_unchanged(workspace, snapshot_archive: Path, expected_manifest: dict, scratch_archive: Path) -> dict:
    """Detect and restore changes to existing tests, removing newly added test files too."""
    errors = []
    try:
        captured = workspace.execute_command(_REPOSITORY_TEST_CAPTURE_COMMAND, timeout=120)
        if captured.exit_code != 0:
            raise RuntimeError(captured.stderr or captured.stdout or "could not capture repository tests")
        download = workspace.file_download(
            source_path="/tmp/openhands-repository-tests.tar.gz",
            destination_path=str(scratch_archive),
        )
        if not download.success:
            raise RuntimeError(f"could not download repository tests: {download.error}")
        actual_manifest = repository_test_manifest_from_archive(scratch_archive)
    except Exception as exc:
        actual_manifest = {}
        errors.append(f"Could not verify repository tests: {type(exc).__name__}: {exc}")
    changed_paths = sorted(
        path for path in set(expected_manifest) | set(actual_manifest)
        if expected_manifest.get(path) != actual_manifest.get(path)
    )
    unchanged = not errors and not changed_paths
    restored = False
    if not unchanged:
        upload = workspace.file_upload(
            source_path=str(snapshot_archive),
            destination_path="/tmp/openhands-repository-tests-trusted.tar.gz",
        )
        if not upload.success:
            errors.append(f"Could not restore trusted repository tests: {upload.error}")
        else:
            restore = workspace.execute_command(
                _REPOSITORY_TEST_DELETE_COMMAND
                + " && tar --no-same-owner --same-permissions "
                "-xzf /tmp/openhands-repository-tests-trusted.tar.gz -C /workspace",
                timeout=120,
            )
            if restore.exit_code != 0:
                errors.append(f"Could not restore repository tests: {restore.stderr or restore.stdout}")
            else:
                try:
                    restored_manifest = capture_repository_tests_snapshot(workspace, scratch_archive)
                    restored = restored_manifest == expected_manifest
                    if not restored:
                        errors.append("Restored repository-test snapshot does not match the trusted snapshot.")
                except Exception as exc:
                    errors.append(f"Could not verify restored repository tests: {type(exc).__name__}: {exc}")
    return {"unchanged": unchanged, "changed_paths": changed_paths, "errors": errors, "restored": restored}


def ensure_generated_tests_unchanged(
    workspace,
    snapshot_archive: Path,
    expected_manifest: dict,
    scratch_archive: Path,
    allow_additions: bool = False,
) -> dict:
    """Compare workspace tests with the host snapshot and restore tampering."""
    current_archive = "/tmp/openhands-generated-tests-current.tar.gz"
    errors = []
    actual_manifest = None
    try:
        captured = workspace.execute_command(
            f"rm -f {current_archive} && "
            f"tar -czf {current_archive} -C /workspace tests/generated",
            timeout=120,
        )
        if captured.exit_code != 0:
            raise RuntimeError(captured.stderr or captured.stdout or "could not capture generated tests")
        download = workspace.file_download(
            source_path=current_archive,
            destination_path=str(scratch_archive),
        )
        if not download.success:
            raise RuntimeError(f"could not download current generated tests: {download.error}")
        actual_manifest = generated_tests_manifest_from_archive(scratch_archive)
    except Exception as exc:
        errors.append(f"Could not verify generated tests: {type(exc).__name__}: {exc}")

    actual_manifest = actual_manifest or {}
    changed_paths = sorted(
        path for path in set(expected_manifest) | set(actual_manifest)
        if expected_manifest.get(path) != actual_manifest.get(path)
        and not (allow_additions and path not in expected_manifest)
    )
    unchanged = not errors and not changed_paths
    if not allow_additions:
        unchanged = unchanged and actual_manifest == expected_manifest
    restored = False
    if not unchanged:
        upload = workspace.file_upload(
            source_path=str(snapshot_archive),
            destination_path="/tmp/openhands-generated-tests-snapshot.tar.gz",
        )
        if not upload.success:
            errors.append(f"Could not restore trusted generated tests: {upload.error}")
        else:
            restore = workspace.execute_command(
                "rm -rf /workspace/tests/generated && "
                "tar --no-same-owner --same-permissions "
                "-xzf /tmp/openhands-generated-tests-snapshot.tar.gz -C /workspace",
                timeout=120,
            )
            if restore.exit_code == 0:
                recapture = workspace.execute_command(
                    f"rm -f {current_archive} && "
                    f"tar -czf {current_archive} -C /workspace tests/generated",
                    timeout=120,
                )
                redownload = workspace.file_download(
                    source_path=current_archive,
                    destination_path=str(scratch_archive),
                ) if recapture.exit_code == 0 else None
                if redownload and redownload.success:
                    try:
                        restored_manifest = generated_tests_manifest_from_archive(scratch_archive)
                        restored = restored_manifest == expected_manifest
                        if not restored:
                            restore_mismatches = sorted(
                                path for path in set(expected_manifest) | set(restored_manifest)
                                if expected_manifest.get(path) != restored_manifest.get(path)
                            )
                            errors.append(
                                "Restored generated-test snapshot differs at: "
                                + ", ".join(restore_mismatches)
                            )
                    except Exception as exc:
                        errors.append(f"Could not verify restored generated tests: {type(exc).__name__}: {exc}")
                if not restored:
                    errors.append("Trusted generated-test snapshot restoration could not be verified.")
            else:
                errors.append(f"Could not restore trusted generated tests: {restore.stderr or restore.stdout}")
    return {
        "unchanged": unchanged,
        "changed_paths": changed_paths,
        "errors": errors,
        "restored": restored,
    }


def load_agent_report(path: Path, attempts: list[dict]) -> dict:
    """Load and cross-check agent diagnoses against captured command output."""
    errors: list[str] = []
    try:
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("report exceeds 4 MiB")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("report root must be an object")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raw = {}
        errors.append(f"Invalid or unavailable agent report: {exc}")

    actual_by_attempt = {item["attempt"]: item for item in attempts}
    required_fields = {
        "schema_version", "detected_stack", "environment", "test_commands",
        "failures", "dependency_installation_failures", "root_causes",
        "modified_files", "proposed_patches", "verified_test_results", "blockers",
    }
    for field in sorted(required_fields - raw.keys()):
        errors.append(f"missing required field: {field}")
    if raw.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    detected_stack = raw.get("detected_stack", {})
    if not isinstance(detected_stack, dict):
        errors.append("detected_stack must be an object")
        detected_stack = {}
    environment = raw.get("environment", {})
    if not isinstance(environment, dict):
        errors.append("environment must be an object")
        environment = {}
    if environment.get("status") not in {"ready", "blocked"}:
        errors.append("environment.status must be ready or blocked")
        environment["status"] = "unknown"

    def checked_list(key):
        value = raw.get(key, [])
        if not isinstance(value, list):
            errors.append(f"{key} must be a list")
            return []
        return value

    diagnoses = []
    raw_failures = raw.get("failures", [])
    if not isinstance(raw_failures, list):
        errors.append("failures must be a list")
        raw_failures = []

    for item in raw_failures:
        if not isinstance(item, dict):
            errors.append("Ignored malformed failure diagnosis")
            continue
        number = item.get("attempt")
        actual = (
            actual_by_attempt.get(number)
            if isinstance(number, int) and not isinstance(number, bool)
            else None
        )
        output = actual.get("output", "") if actual else ""
        category = item.get("category")
        evidence = item.get("evidence")
        if not isinstance(category, str) or category not in FAILURE_CATEGORIES:
            errors.append(f"Attempt {number}: invalid failure category")
            category = "unknown"
        if not isinstance(evidence, list) or not evidence or any(
            not isinstance(quote, str) or not quote.strip() or quote not in output
            for quote in evidence
        ):
            errors.append(f"Attempt {number}: diagnosis evidence does not match captured output")
            category = "unknown"
            evidence = []
        root_cause = item.get("root_cause", "")
        if not isinstance(root_cause, str) or not root_cause.strip():
            errors.append(f"Attempt {number}: diagnosis has no root cause")
            category = "unknown"
            root_cause = ""
        if actual is None or actual.get("exit_code") == 0:
            errors.append(f"Attempt {number}: diagnosis does not match a failed test run")
            category = "unknown"
        diagnoses.append({
            "attempt": number,
            "category": category,
            "evidence": evidence,
            "root_cause": root_cause,
            "repair_action": item.get("repair_action", "") if isinstance(item.get("repair_action"), str) else "",
        })
    diagnosed_attempts = {
        item["attempt"] for item in diagnoses
        if isinstance(item.get("attempt"), int) and not isinstance(item.get("attempt"), bool)
    }
    for actual in attempts:
        if actual.get("exit_code") not in (None, 0) and actual.get("attempt") not in diagnosed_attempts:
            errors.append(f"Attempt {actual.get('attempt')}: missing structured diagnosis")

    dependency_failures = raw.get("dependency_installation_failures", [])
    if not isinstance(dependency_failures, list):
        errors.append("dependency_installation_failures must be a list")
        dependency_failures = []
    verified_dependency_failures = []
    for item in dependency_failures:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"].strip()
        ):
            errors.append("Ignored malformed dependency installation failure")
            continue
        output = item.get("output", "")
        code = item.get("exit_code")
        manager = item.get("manager")
        command = item.get("command")
        if (
            not isinstance(output, str) or not output.strip()
            or not isinstance(code, int) or isinstance(code, bool) or code == 0
            or not isinstance(manager, str) or not manager.strip()
            or not isinstance(command, str) or not command.strip()
        ):
            errors.append(f"Dependency {item['name']}: missing failed command evidence")
            continue
        verified_dependency_failures.append({
            "name": item["name"],
            "manager": manager,
            "command": command,
            "exit_code": code,
            "output": output[-12000:],
            "category": "dependency",
        })

    modified_files = checked_list("modified_files")
    safe_modified_files = []
    for filename in modified_files:
        if not isinstance(filename, str) or Path(filename).is_absolute() or ".." in Path(filename).parts:
            errors.append("Ignored unsafe modified file path")
        else:
            safe_modified_files.append(filename)

    verified = checked_list("verified_test_results")
    test_commands = checked_list("test_commands")
    proposed_patches = checked_list("proposed_patches")
    blockers = checked_list("blockers")

    structured_root_causes = [
        {
            "attempt": item["attempt"],
            "category": item["category"],
            "cause": item["root_cause"],
            "evidence": item["evidence"],
        }
        for item in diagnoses
        if item["category"] != "unknown" and item["root_cause"]
    ]
    structured_root_causes.extend({
        "category": "dependency",
        "cause": f"Could not install {item['name']}",
        "evidence": item["output"][-1000:],
    } for item in verified_dependency_failures)

    return {
        "schema_version": 1,
        "detected_stack": detected_stack,
        "environment": environment,
        "test_commands": test_commands,
        "failures": diagnoses,
        "dependency_installation_failures": verified_dependency_failures,
        "root_causes": structured_root_causes,
        "modified_files": safe_modified_files,
        "proposed_patches": proposed_patches,
        "verified_test_results": verified,
        "blockers": blockers,
        "report_validation_errors": errors,
    }


def final_failure_category(report: dict, attempts: list[dict], environment_ready: bool) -> str | None:
    """Return only a diagnosis supported by captured output or install evidence."""
    if attempts:
        final = attempts[-1]
        if final.get("exit_code") == 0:
            return None
        diagnoses = [
            item for item in report["failures"]
            if item["attempt"] == final["attempt"]
        ]
        categories = {item["category"] for item in diagnoses}
        return categories.pop() if len(categories) == 1 else "unknown"
    if not environment_ready and report["dependency_installation_failures"]:
        return "dependency"
    return "environment"


def update_validation_marker(marker: Path, report: dict) -> bool:
    """Create the marker only after a report has no structural/evidence errors."""
    errors = report.get("report_validation_errors")
    if isinstance(errors, list) and not errors:
        marker.touch()
        return True
    marker.unlink(missing_ok=True)
    return False


def files_in_unified_diff(diff_text: str) -> list[str]:
    """Extract workspace-relative changed paths from captured unified diff headers."""
    paths = set()
    previous_old_path = None
    for line in diff_text.splitlines():
        if line.startswith("--- "):
            fields = shlex.split(line)
            old_path = fields[1].split("\t", 1)[0] if len(fields) > 1 else ""
            if old_path.startswith("a/"):
                previous_old_path = old_path.removeprefix("a/")
            else:
                previous_old_path = None
            continue
        if line.startswith("+++ "):
            fields = shlex.split(line)
            new_path = fields[1].split("\t", 1)[0] if len(fields) > 1 else ""
        else:
            continue
        if new_path.startswith("b/"):
            path = new_path.removeprefix("b/")
        elif new_path.startswith("/workspace/"):
            path = new_path.removeprefix("/workspace/")
        elif new_path == "/dev/null" and previous_old_path:
            path = previous_old_path
        else:
            continue
        if path and ".." not in Path(path).parts:
            paths.add(path)
    return sorted(paths)


def normalize_unified_git_patch(
    diff_text: str, original_root: str, modified_root: str
) -> str:
    """Convert GNU recursive diff output into a Git-apply-compatible patch."""
    old_root = original_root.rstrip("/")
    new_root = modified_root.rstrip("/")
    result = []
    current_path = None
    diff_header_index = None
    old_header_index = None
    new_header_index = None
    file_mode_added = False

    def git_path(path: str) -> str:
        if any(char.isspace() or char in '"\\' for char in path):
            return json.dumps(path, ensure_ascii=False)
        return path

    def relative(path: str, prefix: str) -> str:
        if path == "/dev/null":
            return path
        if not path.startswith(prefix + "/"):
            raise ValueError(f"diff path is outside expected root: {path}")
        name = path[len(prefix) + 1:]
        if not name or ".." in Path(name).parts:
            raise ValueError(f"unsafe diff path: {path}")
        return name

    for line in diff_text.splitlines():
        if line.startswith("diff -ruN "):
            remainder = line[len("diff -ruN "):]
            paths = shlex.split(remainder)
            if len(paths) != 2:
                raise ValueError("unrecognized recursive diff header")
            old_path, new_path = paths
            old_name = relative(old_path, old_root) if old_path != "/dev/null" else old_path
            new_name = relative(new_path, new_root) if new_path != "/dev/null" else new_path
            current_path = new_name if old_name == "/dev/null" else old_name
            if current_path == "/dev/null":
                raise ValueError("diff entry has no repository path")
            diff_header_index = len(result)
            old_header_index = None
            new_header_index = None
            file_mode_added = False
            result.append(
                f"diff --git {git_path('a/' + current_path)} {git_path('b/' + current_path)}"
            )
        elif line.startswith("--- "):
            fields = shlex.split(line)
            old = fields[1] if len(fields) > 1 else ""
            name = relative(old, old_root) if old != "/dev/null" else old
            old_header_index = len(result)
            result.append("--- /dev/null" if name == "/dev/null" else f"--- {git_path('a/' + name)}")
        elif line.startswith("+++ "):
            fields = shlex.split(line)
            new = fields[1] if len(fields) > 1 else ""
            name = relative(new, new_root) if new != "/dev/null" else new
            new_header_index = len(result)
            result.append("+++ /dev/null" if name == "/dev/null" else f"+++ {git_path('b/' + name)}")
        elif line.startswith("@@ ") and old_header_index is not None and new_header_index is not None:
            if line.startswith("@@ -0,0 +"):
                result[old_header_index] = "--- /dev/null"
                if diff_header_index is not None and not file_mode_added:
                    result.insert(diff_header_index + 1, "new file mode 100644")
                    old_header_index += 1
                    new_header_index += 1
                    file_mode_added = True
            elif re.match(r"@@ -\d+(?:,\d+)? \+0(?:,0)? @@", line):
                result[new_header_index] = "+++ /dev/null"
                if diff_header_index is not None and not file_mode_added:
                    result.insert(diff_header_index + 1, "deleted file mode 100644")
                    old_header_index += 1
                    new_header_index += 1
                    file_mode_added = True
            result.append(line)
        elif line.startswith("Binary files "):
            raise ValueError("binary application changes cannot be represented by this text patch generator")
        elif line.startswith("Only in "):
            continue
        else:
            result.append(line)
    return "\n".join(result) + ("\n" if result else "")


def ensure_failure_artifacts(
    output_dir: Path,
    error_name: str,
    error_traceback: str | None = None,
) -> tuple[Path, Path]:
    """Write fallback report and test archive for every exceptional exit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_marker = output_dir / ".report-validated"
    marker_exists = validation_marker.is_file()
    validation_marker.unlink(missing_ok=True)
    report_path = output_dir / "openhands-report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            report = {}
    except (OSError, json.JSONDecodeError):
        report = {}
    report_was_validated = (
        marker_exists
        and report.get("report_valid") is True
        and report.get("report_validation_errors") == []
    )
    report.setdefault("harness_result", report.get("result", "not run"))
    if not isinstance(report.get("repair_history"), list):
        report["repair_history"] = []
    report.update({
        "schema_version": 1,
        "result": "failed",
        "workflow_error": error_name,
        "workflow_traceback": error_traceback,
        "compatibility": {
            "claim": None,
            "note": "Cross-ecosystem compatibility has not been established.",
        },
    })
    for key in (
        "failures", "root_causes", "test_commands",
        "dependency_installation_failures", "modified_files",
        "proposed_patches", "verified_test_results", "blockers",
    ):
        if not isinstance(report.get(key), list):
            report[key] = []
    if not isinstance(report.get("detected_stack"), dict):
        report["detected_stack"] = {}
    if not report_was_validated:
        limitations = report.get("limitations", [])
        report["report_validation_errors"] = [
            "Agent report was not validated against captured test output."
        ]
        report["failures"] = []
        report["root_causes"] = []
        report["dependency_installation_failures"] = []
        report["modified_files"] = []
        report["proposed_patches"] = []
        report["final_failure_category"] = "environment"
        report["blockers"].append(
            "Execution ended before the agent report could be validated."
        )
        if isinstance(limitations, list):
            report["blockers"].extend(
                item for item in limitations if isinstance(item, str)
            )
    environment = report.get("environment")
    if not isinstance(environment, dict):
        environment = {}
    if environment.get("status") not in {"ready", "blocked"}:
        environment["status"] = "blocked"
    report["environment"] = environment
    if not report.get("final_failure_category"):
        report["final_failure_category"] = "environment"
    report["workflow_failure_category"] = "environment"

    patch_path = output_dir / "openhands-application.patch"
    if patch_path.exists():
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        reported_files = report.get("modified_files", [])
        if not isinstance(reported_files, list):
            reported_files = []
        report["modified_files"] = sorted(
            {item for item in reported_files if isinstance(item, str)}
            | set(files_in_unified_diff(patch_text))
        )
        if patch_text.strip():
            report["proposed_patches"] = [{
                "file": "openhands-application.patch",
                "description": "Captured sandbox diff for developer review.",
            }]
            report["application_patch_artifact"] = patch_path.name
        else:
            report["proposed_patches"] = []
            report.pop("application_patch_artifact", None)
    else:
        report["proposed_patches"] = []
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not (output_dir / "openhands-application.patch").exists():
        (output_dir / "openhands-application.patch").write_text("", encoding="utf-8")
    repair_history = report.get("repair_history", [])
    if not isinstance(repair_history, list):
        repair_history = []
    (output_dir / "openhands-repair-history.json").write_text(
        json.dumps(repair_history, indent=2), encoding="utf-8"
    )
    (output_dir / "openhands-test-results.json").write_text(json.dumps({
        "result": report.get("harness_result", report.get("result", "failed")),
        "final_exit_code": report.get("harness_final_exit_code"),
        "validated_test_command": report.get("validated_test_command"),
        "environment_ready": report.get("environment_ready", False),
        "attempts": report.get("harness_attempts", []),
        "verified_test_results": report.get("verified_test_results", []),
    }, indent=2), encoding="utf-8")

    archive_path = output_dir / "generated-tests.tar.gz"
    archive_valid = False
    if archive_path.exists():
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.getmembers()
            archive_valid = True
        except (OSError, tarfile.TarError):
            pass
    if not archive_valid:
        with tarfile.open(archive_path, "w:gz") as archive:
            directory = tarfile.TarInfo("tests/generated/")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
    return report_path, archive_path


_INFRASTRUCTURE_FAILURE_RE = re.compile(
    r"ModuleNotFoundError|No module named|Cannot find module|ERR_MODULE_NOT_FOUND|"
    r"Could not resolve dependencies|Could not resolve package|No matching distribution|"
    r"Could not find a version that satisfies|command not found|"
    r"Connection refused|ECONNREFUSED|ETIMEDOUT|ServiceUnavailable|"
    r"Could not connect|Failed to download|Failed to resolve plugin",
    re.I,
)
_TEST_FAILURE_EVIDENCE_RE = re.compile(
    r"(?im)^\s*(?:FAILED\b|FAIL:\s|FAIL\s)|"
    r"AssertionError|AssertionFailedError|AssertionError|"
    r"Expected:|Received:|expected .{0,100}(?:got|received|actual)",
)
_PYTEST_NODE_RE = re.compile(r"(?im)^\s*(?:FAILED|ERROR)\s+([^\s]+::[^\s]+)")
_JEST_TEST_RE = re.compile(r"(?m)^\s*●\s*(.+?)\s*$")
_JUNIT_TEST_RE = re.compile(r"(?im)^\s*([\w.$]+\([^)]*\))\s+<<<\s+FAILURE!")
_GO_TEST_RE = re.compile(r"(?m)^--- FAIL:\s+([^\s(]+)")
_MOCHA_TEST_RE = re.compile(r"(?m)^\s*\d+\)\s+(.+?)\s*$")
_GRADLE_TEST_RE = re.compile(r"(?m)^\s*(.+?)\s+>\s+(.+?)\s+FAILED\s*$")


def _failing_test_identifiers(output: str) -> set[str]:
    identifiers = set(_PYTEST_NODE_RE.findall(output))
    identifiers.update(_JEST_TEST_RE.findall(output))
    identifiers.update(_JUNIT_TEST_RE.findall(output))
    identifiers.update(_GO_TEST_RE.findall(output))
    identifiers.update(_MOCHA_TEST_RE.findall(output))
    identifiers.update(f"{suite} > {test}" for suite, test in _GRADLE_TEST_RE.findall(output))
    return {item.strip() for item in identifiers if item.strip()}


def validate_repair_diagnosis(report_text: str, attempt: int, output: str) -> dict:
    """Validate one agent diagnosis and decide whether app repair is justified."""
    result = {
        "attempt": attempt,
        "reported_category": "unknown",
        "category": "unknown",
        "evidence": [],
        "root_cause": "",
        "repair_action": "",
        "validated": False,
        "repair_allowed": False,
        "decision_reason": "No valid structured diagnosis was available; human review required.",
    }
    try:
        if len(report_text) > 4 * 1024 * 1024:
            raise ValueError("diagnosis report exceeds 4 MiB")
        raw = json.loads(report_text)
        if not isinstance(raw, dict) or raw.get("schema_version") != 1 or not isinstance(raw.get("failures"), list):
            raise ValueError("report must be a schema-version-1 object with a failures list")
        matching = [
            item for item in raw["failures"]
            if isinstance(item, dict)
            and isinstance(item.get("attempt"), int)
            and not isinstance(item.get("attempt"), bool)
            and item.get("attempt") == attempt
        ]
        if len(matching) != 1:
            raise ValueError("report must contain exactly one diagnosis for this attempt")
        item = matching[0]
        category = item.get("category")
        failing_test = item.get("failing_test")
        evidence = item.get("evidence")
        root_cause_evidence = item.get("root_cause_evidence")
        root_cause = item.get("root_cause")
        if not isinstance(category, str) or category not in FAILURE_CATEGORIES:
            raise ValueError("diagnosis category is invalid")
        if not isinstance(evidence, list) or not evidence or any(
            not isinstance(quote, str) or not quote.strip() or quote not in output
            for quote in evidence
        ):
            raise ValueError("diagnosis evidence does not exactly match captured test output")
        actual_tests = _failing_test_identifiers(output)
        if not isinstance(failing_test, str) or failing_test not in actual_tests:
            raise ValueError("diagnosis does not identify a failing test present in captured output")
        if failing_test not in evidence:
            raise ValueError("diagnosis evidence must quote the actual failing test identifier")
        if not isinstance(root_cause_evidence, list) or not root_cause_evidence or any(
            not isinstance(quote, str) or not quote.strip() or quote not in output or quote not in evidence
            for quote in root_cause_evidence
        ):
            raise ValueError("root-cause evidence must quote captured output and appear in diagnosis evidence")
        if not isinstance(root_cause, str) or not root_cause.strip():
            raise ValueError("diagnosis has no root cause")
        result.update({
            "reported_category": category,
            "category": category,
            "evidence": evidence,
            "failing_test": failing_test,
            "root_cause_evidence": root_cause_evidence,
            "root_cause": root_cause,
            "repair_action": item.get("repair_action", "") if isinstance(item.get("repair_action"), str) else "",
            "validated": True,
            "decision_reason": "Diagnosis has exact output evidence and a root cause.",
        })
    except (ValueError, json.JSONDecodeError) as exc:
        result["decision_reason"] = f"Diagnosis rejected: {exc}; human review required."
        return result

    if result["category"] not in {"application bug", "regression"}:
        result["decision_reason"] = (
            f"Failure classified as {result['category']}; application-code repair was not attempted."
        )
        return result
    if _INFRASTRUCTURE_FAILURE_RE.search(output):
        result["category"] = "unknown"
        result["decision_reason"] = (
            "Captured output contains infrastructure/dependency evidence that conflicts "
            "with an application-failure classification; human review required."
        )
        return result
    if not _TEST_FAILURE_EVIDENCE_RE.search(output):
        result["category"] = "unknown"
        result["decision_reason"] = (
            "Captured output does not contain a test assertion/failure signature; "
            "application repair was not attempted; human review required."
        )
        return result
    result["repair_allowed"] = True
    result["decision_reason"] = "Verified application bug/regression diagnosis; sandbox repair is allowed."
    return result


def _check_generated_tests(test_guard, stage: str) -> dict:
    if test_guard is None:
        return {"unchanged": True, "changed_paths": [], "errors": [], "restored": False}
    try:
        result = test_guard()
        if isinstance(result, dict):
            return result
        return {"unchanged": bool(result), "changed_paths": [], "errors": [], "restored": False}
    except Exception as exc:
        return {
            "unchanged": False,
            "changed_paths": [],
            "errors": [f"Generated-test integrity check failed at {stage}: {type(exc).__name__}: {exc}"],
            "restored": False,
        }


def _read_repair_diagnosis(workspace, attempt: int, output: str) -> dict:
    """Read and strictly validate the current agent report for one attempt."""
    report_check = workspace.execute_command(
        "test -f /workspace/openhands-report.json && "
        "test ! -L /workspace/openhands-report.json",
        timeout=30,
    )
    if report_check.exit_code != 0:
        raise ValueError("agent report is missing or not a regular file")
    report_size = workspace.execute_command(
        "wc -c < /workspace/openhands-report.json", timeout=30
    )
    if report_size.exit_code != 0 or not (report_size.stdout or "").strip().isdigit():
        raise ValueError("could not validate agent report size")
    if int(report_size.stdout.strip()) > 4 * 1024 * 1024:
        raise ValueError("agent report exceeds 4 MiB")
    diagnosis_result = workspace.execute_command(
        "cat -- /workspace/openhands-report.json", timeout=30
    )
    if diagnosis_result.exit_code != 0:
        raise ValueError(diagnosis_result.stderr or "agent report unavailable")
    return validate_repair_diagnosis(diagnosis_result.stdout or "", attempt, output)


def run_repair_loop(
    workspace,
    conversation,
    test_command: "str | TestExecutionPlan",
    max_repairs: int,
    generated_tests_guard=None,
):
    """Run full-suite attempts from one execution plan.

    Production passes a ``TestExecutionPlan``. Every attempt then renders the
    suite through ``sandbox_test_command`` using the same interpreter and
    environment, so a repair cannot silently change how the suite is executed.
    A raw string remains accepted for legacy callers/tests only.
    """
    if isinstance(test_command, TestExecutionPlan):
        execution_plan = test_command
        test_command = sandbox_test_command(execution_plan)
        plan_metadata = {
            "plan_digest": execution_plan.digest,
            "execution_plan": execution_plan.to_dict(),
        }
    else:
        execution_plan = None
        test_command = str(test_command)
        plan_metadata = {}
        legacy_tokens = shlex.split(test_command)
        if legacy_tokens and _python_runner(legacy_tokens):
            raise EnvironmentResolutionError(
                "Python suite execution requires a TestExecutionPlan; a raw "
                "command string has no verified interpreter.",
                evidence={"command": test_command},
            )
    max_repairs = max(0, min(int(max_repairs), 3))
    attempts = []
    final_result = None
    pending_repair_index = None
    for attempt in range(max_repairs + 1):
        integrity = _check_generated_tests(generated_tests_guard, "before test run")
        if not integrity.get("unchanged"):
            if pending_repair_index is not None:
                attempts[pending_repair_index]["repair_status"] = "repair_not_verified_test_integrity_failed"
            attempts.append({
                "attempt": attempt + 1,
                "command": test_command,
                **plan_metadata,
                "exit_code": None,
                "output": "",
                "test_integrity": integrity,
                "execution_error": "GeneratedTestIntegrityError",
                "repair_decision": "human_review",
                "repair_decision_reason": "Generated-test integrity verification failed before suite execution.",
            })
            final_result = SimpleNamespace(
                exit_code=1,
                stdout="Repository tests changed; trusted snapshots were restored where possible.",
                stderr="Suite was not run because test integrity verification failed.",
            )
            break
        try:
            result = workspace.execute_command(test_command, timeout=1200)
        except Exception as exc:
            if pending_repair_index is not None:
                attempts[pending_repair_index]["repair_status"] = "repair_not_verified_test_execution_failed"
            attempts.append({
                "attempt": attempt + 1,
                "command": test_command,
                **plan_metadata,
                "exit_code": None,
                "output": "",
                "execution_error": type(exc).__name__,
            })
            final_result = SimpleNamespace(
                exit_code=1, stdout="", stderr="Test command execution failed."
            )
            break
        output = (result.stdout or "") + (result.stderr or "")
        attempts.append({
            "attempt": attempt + 1,
            "command": test_command,
            **plan_metadata,
            "exit_code": result.exit_code,
            "output": output[-12000:],
        })
        final_result = result
        test_integrity = _check_generated_tests(generated_tests_guard, "after test run")
        attempts[-1]["test_integrity"] = test_integrity
        if not test_integrity.get("unchanged"):
            attempts[-1]["output"] += "\nHarness: test files changed during execution; run result is invalid."
            attempts[-1]["exit_code"] = 1
            attempts[-1]["repair_decision"] = "human_review"
            if pending_repair_index is not None:
                attempts[pending_repair_index]["repair_status"] = "repair_not_verified_test_integrity_failed"
                pending_repair_index = None
            attempts[-1]["repair_decision_reason"] = "Generated tests changed during test execution; the run result was invalidated."
            final_result = SimpleNamespace(
                exit_code=1,
                stdout=output,
                stderr="Test files changed during execution; trusted snapshots were restored. No repair was attempted.",
            )
            break
        if pending_repair_index is not None:
            attempts[pending_repair_index]["repair_status"] = (
                "repair_verified" if result.exit_code == 0 else "repair_not_verified"
            )
            pending_repair_index = None
        if result.exit_code == 0:
            break

        # Diagnosis and repair are separate turns. Unclassified failures never
        # authorize an application edit.
        try:
            conversation.send_message(f"""
The complete test command failed on attempt {attempt + 1} of {max_repairs + 1}.

Actual output follows:
{output[-12000:]}

Diagnose only. Do not modify, create, delete, rename, or chmod any files.
In /workspace/openhands-report.json, update the structured diagnosis for this
attempt with category, failing_test (an exact test identifier visible in this
output), exact evidence that includes that identifier, root_cause, and
root_cause_evidence (the exact output text supporting that cause), plus
repair_action. Use unknown if the cause is uncertain or no exact test
identifier is available. Do not claim the suite
passed. Do not classify dependency, environment, build or invalid-test
failures as application bugs.
""")
            conversation.run()
        except Exception as exc:
            attempts[-1]["diagnosis_error"] = type(exc).__name__
            attempts[-1]["repair_decision"] = "human_review"
            attempts[-1]["repair_decision_reason"] = "Agent diagnosis turn failed."
            break

        diagnosis_integrity = _check_generated_tests(generated_tests_guard, "after diagnosis")
        attempts[-1]["diagnosis_test_integrity"] = diagnosis_integrity
        if not diagnosis_integrity.get("unchanged"):
            attempts[-1]["repair_decision"] = "human_review"
            attempts[-1]["repair_decision_reason"] = "Test files changed during diagnosis; repair was blocked."
            break

        try:
            diagnosis = _read_repair_diagnosis(workspace, attempt + 1, output)
        except Exception as exc:
            diagnosis = validate_repair_diagnosis("", attempt + 1, output)
            diagnosis["decision_reason"] = f"Could not validate diagnosis: {type(exc).__name__}: {exc}; human review required."

        # A malformed or incomplete structured report should not end the
        # repair flow immediately. Give OpenHands one report-only correction
        # turn, then apply the same strict validation gates again.
        if not diagnosis["validated"]:
            attempts[-1]["diagnosis_validation_error"] = diagnosis["decision_reason"]
            try:
                conversation.send_message(f"""
The harness rejected the diagnosis report for attempt {attempt + 1}:
{diagnosis['decision_reason']}

Replace the complete `failures` entry for attempt {attempt + 1} in
/workspace/openhands-report.json with a complete entry containing every
required schema field: `attempt`, `category`, `failing_test` (an exact test
identifier present below), `evidence` containing that identifier and exact
output text, `root_cause`, `root_cause_evidence` (exact output text also
included in `evidence`), and `repair_action`. Preserve every other entry in
`failures` and all other report fields and metadata. Do not modify, create,
delete, rename, or chmod any other file. Use `unknown` if the evidence does
not support an application bug or regression. Do not attempt an application
repair in this turn.

Captured test output:
{output[-12000:]}
""")
                conversation.run()
                retry_integrity = _check_generated_tests(
                    generated_tests_guard, "after diagnosis correction"
                )
                attempts[-1]["diagnosis_correction_test_integrity"] = retry_integrity
                if not retry_integrity.get("unchanged"):
                    attempts[-1]["diagnosis_recovery_error"] = (
                        "Test files changed during diagnosis correction; repair was blocked."
                    )
                    diagnosis = validate_repair_diagnosis("", attempt + 1, output)
                else:
                    diagnosis = _read_repair_diagnosis(workspace, attempt + 1, output)
                    attempts[-1]["diagnosis_correction"] = "validated" if diagnosis["validated"] else "rejected"
            except Exception as exc:
                attempts[-1]["diagnosis_recovery_error"] = f"{type(exc).__name__}: {exc}"
                diagnosis = validate_repair_diagnosis("", attempt + 1, output)
                diagnosis["decision_reason"] = (
                    f"Diagnosis correction failed: {type(exc).__name__}: {exc}; human review required."
                )
        attempts[-1]["diagnosis"] = diagnosis

        if not diagnosis["repair_allowed"]:
            attempts[-1]["repair_decision"] = "human_review"
            attempts[-1]["repair_decision_reason"] = diagnosis["decision_reason"]
            break
        if attempt == max_repairs:
            attempts[-1]["repair_decision"] = "limit_reached"
            attempts[-1]["repair_decision_reason"] = "Verified application issue, but the repair limit was reached."
            break

        attempts[-1]["repair_decision"] = "repair_authorized"
        try:
            conversation.send_message(f"""
The harness validated this diagnosis as an application bug/regression using
exact captured test output:
{json.dumps(diagnosis, ensure_ascii=False)}

Make at most one minimal correction to application source code inside this
Docker workspace. Do not modify, delete, rename, chmod, skip, or weaken any
generated or existing tests. Do not edit workflow or trusted automation files,
dependency manifests, lockfiles, or test configuration. Never access or print
credentials. Record the changed application files and proposed patch in
/workspace/openhands-report.json. Do not claim success; the harness will verify
generated-test integrity and rerun the complete validated suite.
""")
            conversation.run()
            attempts[-1]["repair_status"] = "repair_attempted_pending_suite_verification"
        except Exception as exc:
            attempts[-1]["repair_status"] = "agent_turn_failed"
            attempts[-1]["repair_error"] = type(exc).__name__
            attempts[-1]["repair_decision"] = "human_review"
            attempts[-1]["repair_decision_reason"] = (
                "OpenHands repair turn failed; no repair was verified and the suite was not rerun."
            )

        repair_integrity = _check_generated_tests(generated_tests_guard, "after repair")
        attempts[-1]["repair_test_integrity"] = repair_integrity
        if not repair_integrity.get("unchanged"):
            attempts[-1]["repair_status"] = "rejected_test_files_modified"
            attempts[-1]["repair_decision_reason"] = "Test files changed during repair; trusted snapshots were restored and no rerun was allowed."
            break
        if attempts[-1]["repair_status"] == "agent_turn_failed":
            break
        pending_repair_index = len(attempts) - 1
    return final_result, attempts

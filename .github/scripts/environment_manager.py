"""Trusted Python environment preparation and validation.

All commands are built from validated paths and executed inside the isolated
Docker workspace. This module never executes repository code or installation
hooks, never touches the host environment, and never installs into system
Python. Repairs are explicit operations selected by ``repair_engine``.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from environment_detector import declared_test_requirement  # noqa: E402

MAX_COMMAND_OUTPUT = 12000
DEFAULT_COMMAND_TIMEOUT = 300
DEADLINE_EXIT_CODE = 124
_EXACT_VERSION_RE = re.compile(r"^(\d+)\.(\d+)(?:\.\d+)?$")
_UNSAFE_PATH_RE = re.compile(r'["`$\\\n\r\x00]')
# Only these explicit observations authorize deleting a virtual environment.
_INVALID_VENV_REASONS = {
    "venv path is a symlink",
    "pyvenv.cfg missing",
    "interpreter missing or not executable",
}

_INTROSPECTION = (
    "import json,sys,importlib.util;"
    "print(json.dumps({"
    "'interpreter':sys.executable,"
    "'version':list(sys.version_info[:3]),"
    "'has_pip':bool(importlib.util.find_spec('pip')),"
    "'has_pytest':bool(importlib.util.find_spec('pytest'))"
    "}))"
)


def _result(exit_code, stdout="", stderr=""):
    return SimpleNamespace(exit_code=exit_code, stdout=stdout or "", stderr=stderr or "")


def _redact(text: str) -> str:
    import os

    for name, value in os.environ.items():
        if value and re.search(r"(?:TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|AUTH)", name, re.I):
            text = text.replace(value, "[REDACTED]")
    return text


def _tail(text: str) -> str:
    return _redact(text)[-MAX_COMMAND_OUTPUT:]


def _json_last(output: str) -> dict | None:
    for line in reversed((output or "").splitlines()):
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


class EnvironmentManager:
    """Prepare and validate one project virtual environment in the sandbox."""

    def __init__(self, workspace, venv_path: str = "/workspace/.venv"):
        if (
            not venv_path.startswith("/workspace/")
            or ".." in Path(venv_path).parts
            or _UNSAFE_PATH_RE.search(venv_path)
        ):
            raise ValueError("unsafe virtual environment path")
        self.workspace = workspace
        self.venv_path = venv_path.rstrip("/")
        self.interpreter = f"{self.venv_path}/bin/python"

    def _run(self, command: str, timeout: int | None = None, deadline: float | None = None):
        budget = int(timeout) if timeout is not None else DEFAULT_COMMAND_TIMEOUT
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _result(
                    DEADLINE_EXIT_CODE,
                    stderr="repair deadline exceeded before command execution",
                )
            budget = min(budget, max(1, int(remaining)))
        return self.workspace.execute_command(command, timeout=max(1, budget))

    def venv_status(self, timeout: int | None = None, deadline: float | None = None) -> dict:
        """Inspect the venv; distinguish valid, invalid and inconclusive states."""
        quoted_root = shlex.quote(self.venv_path)
        quoted_python = shlex.quote(self.interpreter)
        command = (
            f"if [ -L {quoted_root} ]; then "
            "echo '{\"valid\":false,\"reason\":\"venv path is a symlink\"}'; "
            f"elif [ ! -f {quoted_root}/pyvenv.cfg ]; then "
            "echo '{\"valid\":false,\"reason\":\"pyvenv.cfg missing\"}'; "
            f"elif [ ! -x {quoted_python} ]; then "
            "echo '{\"valid\":false,\"reason\":\"interpreter missing or not executable\"}'; "
            f"else {quoted_python} -c {shlex.quote(_INTROSPECTION)}; fi"
        )
        result = self._run(command, timeout=timeout, deadline=deadline)
        info = _json_last(result.stdout)
        if not isinstance(info, dict) or result.exit_code != 0:
            # No structured evidence means inconclusive. Never treat this as
            # proof that the venv is invalid.
            return {
                "state": "inconclusive",
                "valid": False,
                "reason": "venv inspection produced no structured result",
                "exit_code": result.exit_code,
                "output": _tail(result.stdout + "\n" + result.stderr),
            }
        info.setdefault("valid", False)
        info["state"] = "valid" if info.get("valid") is True else "invalid"
        info["path"] = self.venv_path
        info["interpreter"] = self.interpreter
        return info

    def _base_interpreter(
        self,
        declared_version: str | None,
        timeout: int | None = None,
        deadline: float | None = None,
    ) -> str:
        if declared_version:
            match = _EXACT_VERSION_RE.match(str(declared_version))
            if match:
                candidate = f"python{match.group(1)}.{match.group(2)}"
                probe = self._run(
                    f"command -v {shlex.quote(candidate)} >/dev/null 2>&1",
                    timeout=timeout, deadline=deadline,
                )
                if probe.exit_code == 0:
                    return candidate
        for candidate in ("python3", "python"):
            probe = self._run(
                f"command -v {shlex.quote(candidate)} >/dev/null 2>&1",
                timeout=timeout, deadline=deadline,
            )
            if probe.exit_code == 0:
                return candidate
        raise RuntimeError("No base Python interpreter is available in the sandbox")

    def ensure_venv(
        self,
        detection: dict | None = None,
        timeout: int | None = None,
        deadline: float | None = None,
    ) -> dict:
        """Create/recreate the venv only on positive evidence of invalidity."""
        status = self.venv_status(timeout=timeout, deadline=deadline)
        if status.get("state") == "valid":
            return {
                "changed": False,
                "status": status,
                "commands": [],
                "reason": "existing virtual environment is valid",
            }
        if status.get("state") != "invalid":
            return {
                "changed": False,
                "status": status,
                "commands": [],
                "reason": "venv inspection was inconclusive; refusing to delete it",
                "blocked": True,
            }
        if status.get("reason") not in _INVALID_VENV_REASONS:
            return {
                "changed": False,
                "status": status,
                "commands": [],
                "reason": "invalid venv reason is not an approved recreation trigger",
                "blocked": True,
            }

        commands = []
        quoted_root = shlex.quote(self.venv_path)
        remove = self._run(
            f"if [ -e {quoted_root} ] || [ -L {quoted_root} ]; then "
            f"rm -rf -- {quoted_root}; fi",
            timeout=timeout, deadline=deadline,
        )
        commands.append({"command": remove_command_repr(self.venv_path), "exit_code": remove.exit_code})
        if remove.exit_code == DEADLINE_EXIT_CODE or remove.exit_code != 0:
            return {
                "changed": False,
                "status": self.venv_status(timeout=timeout, deadline=deadline),
                "commands": commands,
                "reason": "could not safely remove the invalid virtual environment",
                "blocked": True,
            }

        python = self._base_interpreter(
            (detection or {}).get("python", {}).get("declared_python_version"),
            timeout=timeout, deadline=deadline,
        )
        create = self._run(
            f"{shlex.quote(python)} -m venv {quoted_root}",
            timeout=timeout, deadline=deadline,
        )
        commands.append({"command": f"{python} -m venv {self.venv_path}", "exit_code": create.exit_code, "output": _tail(create.stderr or create.stdout)})
        if create.exit_code != 0:
            retry = self._run(
                f"{shlex.quote(python)} -m venv --without-pip {quoted_root}",
                timeout=timeout, deadline=deadline,
            )
            commands.append({"command": f"{python} -m venv --without-pip {self.venv_path}", "exit_code": retry.exit_code, "output": _tail(retry.stderr or retry.stdout)})
            if retry.exit_code != 0:
                return {
                    "changed": False,
                    "status": self.venv_status(timeout=timeout, deadline=deadline),
                    "commands": commands,
                    "reason": "virtual environment creation failed",
                    "blocked": True,
                }

        pip = self.bootstrap_pip(timeout=timeout, deadline=deadline)
        commands.extend(pip["commands"])
        status = self.venv_status(timeout=timeout, deadline=deadline)
        return {
            "changed": True,
            "status": status,
            "commands": commands,
            "reason": "virtual environment created",
            "blocked": not status.get("valid", False),
        }

    def bootstrap_pip(
        self,
        timeout: int | None = None,
        deadline: float | None = None,
    ) -> dict:
        """Bootstrap pip in the project venv using the stdlib ensurepip."""
        quoted_python = shlex.quote(self.interpreter)
        probe = self._run(f"{quoted_python} -m pip --version", timeout=timeout, deadline=deadline)
        commands = []
        if probe.exit_code == 0:
            return {"changed": False, "commands": commands, "reason": "pip already available"}
        bootstrap = self._run(
            f"{quoted_python} -m ensurepip --upgrade", timeout=timeout, deadline=deadline
        )
        commands.append({
            "command": f"{self.interpreter} -m ensurepip --upgrade",
            "exit_code": bootstrap.exit_code,
            "output": _tail(bootstrap.stderr or bootstrap.stdout),
        })
        verify = self._run(
            f"{quoted_python} -m pip --version", timeout=timeout, deadline=deadline
        )
        commands.append({
            "command": f"{self.interpreter} -m pip --version",
            "exit_code": verify.exit_code,
            "output": _tail(verify.stderr or verify.stdout),
        })
        return {
            "changed": verify.exit_code == 0,
            "commands": commands,
            "reason": "pip bootstrapped" if verify.exit_code == 0 else "pip could not be bootstrapped",
            "blocked": verify.exit_code != 0,
        }

    def install_test_framework(
        self,
        detection: dict | None = None,
        timeout: int | None = None,
        deadline: float | None = None,
    ) -> dict:
        """Install pytest into the selected venv only when it is missing."""
        framework = ((detection or {}).get("python") or {}).get("test_framework") or "pytest"
        if framework != "pytest":
            return {
                "changed": False,
                "commands": [],
                "reason": f"framework {framework!r} does not require pytest installation",
                "blocked": False,
            }
        from environment_detector import declared_test_requirements

        pins = declared_test_requirements(detection or {})
        if len(pins) > 1:
            return {
                "changed": False,
                "commands": [],
                "reason": f"conflicting pytest pins declared in repository: {pins}",
                "blocked": True,
            }
        quoted_python = shlex.quote(self.interpreter)
        probe = self._run(
            f"{quoted_python} -c {shlex.quote('import pytest')}",
            timeout=timeout, deadline=deadline,
        )
        if probe.exit_code == 0:
            return {"changed": False, "commands": [], "reason": "pytest already installed"}
        requirement = pins[0] if pins else "pytest"
        command = (
            f"{quoted_python} -m pip install --disable-pip-version-check --no-input "
            f"{shlex.quote(requirement)}"
        )
        result = self._run(command, timeout=timeout, deadline=deadline)
        record = {
            "command": f"{self.interpreter} -m pip install --disable-pip-version-check --no-input {requirement}",
            "exit_code": result.exit_code,
            "output": _tail(result.stderr or result.stdout),
        }
        verify = self._run(
            f"{quoted_python} -c {shlex.quote('import pytest')}",
            timeout=timeout, deadline=deadline,
        )
        return {
            "changed": verify.exit_code == 0,
            "commands": [record],
            "reason": "pytest installed in project venv" if verify.exit_code == 0 else "pytest installation failed",
            "blocked": verify.exit_code != 0,
        }

    def install_dependencies(
        self,
        detection: dict,
        timeout: int | None = None,
        deadline: float | None = None,
    ) -> dict:
        """Install declared dependencies with the repository's package manager."""
        python = (detection or {}).get("python") or {}
        if not python.get("detected"):
            return {"changed": False, "commands": [], "reason": "no Python environment detected", "blocked": False}
        manager = python.get("package_manager")
        requirements = [name for name in python.get("requirements_files", []) if re.fullmatch(r"requirements[\w.-]*\.txt", name)]
        # Detection reads the same sandbox workspace; all install paths are
        # sandbox-absolute and the detector reports evidence.root == /workspace.
        root = "/workspace"
        quoted_python = shlex.quote(self.interpreter)

        if manager in {"poetry", "pipenv"}:
            return {
                "changed": False,
                "commands": [],
                "reason": f"trusted dependency installation for {manager} is not implemented in Phase 1",
                "blocked": True,
            }
        if manager not in {"pip", "uv"}:
            return {
                "changed": False,
                "commands": [],
                "reason": f"unsupported or undeclared package manager: {manager}",
                "blocked": True,
            }
        if not requirements:
            return {
                "changed": False,
                "commands": [],
                "reason": "no trusted, declared requirements files to install",
                "blocked": False,
            }

        commands = []
        for name in sorted(requirements):
            requirement_path = f"{root.rstrip('/')}/{name}"
            quoted_requirement = shlex.quote(requirement_path)
            constraint = f"{root.rstrip('/')}/constraints.txt"
            constraint_probe = self._run(
                f"test -f {shlex.quote(constraint)}", timeout=timeout, deadline=deadline
            )
            constraint_arg = f" -c {shlex.quote(constraint)}" if constraint_probe.exit_code == 0 else ""
            if manager == "uv":
                command = f"uv pip install --python {quoted_python} -r {quoted_requirement}{constraint_arg}"
            else:
                command = f"{quoted_python} -m pip install --disable-pip-version-check --no-input -r {quoted_requirement}{constraint_arg}"
            result = self._run(command, timeout=timeout, deadline=deadline)
            commands.append({
                "command": command,
                "exit_code": result.exit_code,
                "output": _tail(result.stderr or result.stdout),
            })
            if result.exit_code != 0:
                return {
                    "changed": False,
                    "commands": commands,
                    "reason": f"dependency installation failed for {name}",
                    "blocked": True,
                }
        return {"changed": True, "commands": commands, "reason": "declared dependencies installed", "blocked": False}

    def verify(
        self,
        detection: dict | None = None,
        timeout: int | None = None,
        deadline: float | None = None,
    ) -> dict:
        """Independently verify the venv's runtime, pip, and test framework."""
        status = self.venv_status(timeout=timeout, deadline=deadline)
        framework = ((detection or {}).get("python") or {}).get("test_framework") or "pytest"
        framework_ok = True
        if framework == "pytest":
            probe = self._run(
                f"{shlex.quote(self.interpreter)} -c {shlex.quote('import pytest')}",
                timeout=timeout, deadline=deadline,
            )
            framework_ok = probe.exit_code == 0
        return {
            "ok": bool(status.get("valid")) and bool(status.get("has_pip")) and framework_ok,
            "status": status,
            "framework": framework,
            "framework_ok": framework_ok,
        }


def remove_command_repr(venv_path: str) -> str:
    """Render a non-secret representation of the venv removal step."""
    return f"rm -rf -- {venv_path}"

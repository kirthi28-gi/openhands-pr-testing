"""Validation and retry helpers for the OpenHands PR test runner."""

from __future__ import annotations

import json
import re
import shlex
import tarfile
from pathlib import Path
from types import SimpleNamespace


FAILURE_CATEGORIES = {
    "dependency",
    "environment",
    "build",
    "incorrect test",
    "application bug",
    "regression",
    "unknown",
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


def _python_runner(tokens: list[str]) -> bool:
    if not tokens:
        return False
    if tokens[0] in {"pytest", ".venv/bin/pytest", "/workspace/.venv/bin/pytest"}:
        return _full_pytest_arguments(tokens[1:])
    if tokens[0] in {"python", "python3", ".venv/bin/python", "/workspace/.venv/bin/python"}:
        return _python_module_runner(tokens[1:])
    if re.fullmatch(r"python3\.\d+", tokens[0]):
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


def _run_verification_check(workspace, checks: list[dict], name: str, argv: list[str]) -> bool:
    command = shlex.join(argv)
    try:
        result = workspace.execute_command(sandbox_test_command(command), timeout=600)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = (stdout + ("\n" if stdout and stderr else "") + stderr)[-12000:]
        code = result.exit_code
    except Exception as exc:
        code = None
        output = f"Verification command raised {type(exc).__name__}: {exc}"
    passed = code == 0
    checks.append({
        "name": name,
        "command": command,
        "exit_code": code,
        "output": output,
        "passed": passed,
    })
    return passed


def _python_verification_parts(tokens: list[str]) -> tuple[list[str], str, list[str], str]:
    """Return wrapper, interpreter, runner argv, and test framework."""
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
        runner = tokens

    if runner[0] in {"pytest", ".venv/bin/pytest", "/workspace/.venv/bin/pytest"}:
        interpreter = {
            ".venv/bin/pytest": ".venv/bin/python",
            "/workspace/.venv/bin/pytest": "/workspace/.venv/bin/python",
        }.get(runner[0], "python")
        return wrapper, interpreter, [runner[0]], "pytest"
    if runner[0] in {"python", "python3", ".venv/bin/python", "/workspace/.venv/bin/python"} or re.fullmatch(r"python3\.\d+", runner[0]):
        interpreter = runner[0]
        framework = runner[2]
        return wrapper, interpreter, [interpreter, "-m", framework], framework
    raise ValueError("Could not derive the Python runtime and test runner from the validated command")


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


def verify_test_environment(workspace, repository_root: Path, test_command: str) -> dict:
    """Independently check runtime, runner, dependency health and test discovery.

    Collection/listing is recorded separately and is never sufficient by itself
    to set ``ready``. Test functions are not executed by these checks.
    """
    checks: list[dict] = []
    failures: list[str] = []
    limitations: list[str] = []
    ecosystem = "unknown"
    validated = validate_test_command(test_command, repository_root)
    tokens = shlex.split(validated)

    def check(name: str, argv: list[str]) -> bool:
        passed = _run_verification_check(workspace, checks, name, argv)
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
        wrapper, interpreter, runner, framework = _python_verification_parts(tokens)
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
        required_ok = runtime_ok and runner_ok and pip_probe and (pip_check or not pip_available) and collection_ok
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
        "ready": required_ok,
        "checks": checks,
        "failures": failures,
        "blockers": failures.copy(),
        "limitations": limitations,
    }


def sandbox_test_command(command: str) -> str:
    """Build a shell-safe invocation while removing inherited credentials."""
    argv = shlex.split(command)
    return "cd /workspace && " + shlex.join([
        "env", "-u", "SESSION_API_KEY", "-u", "OH_SESSION_API_KEYS_0",
        "-u", "LLM_API_KEY", *argv,
    ])


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


def ensure_failure_artifacts(output_dir: Path, error_name: str) -> tuple[Path, Path]:
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
    report.update({
        "schema_version": 1,
        "result": "failed",
        "workflow_error": error_name,
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


def run_repair_loop(workspace, conversation, test_command: str, max_repairs: int):
    """Run tests and ask the same agent to repair between failed executions."""
    max_repairs = max(0, min(int(max_repairs), 3))
    attempts = []
    final_result = None
    for attempt in range(max_repairs + 1):
        try:
            result = workspace.execute_command(test_command, timeout=1200)
        except Exception as exc:
            attempts.append({
                "attempt": attempt + 1,
                "command": test_command,
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
            "exit_code": result.exit_code,
            "output": output[-12000:],
        })
        final_result = result
        if result.exit_code == 0 or attempt == max_repairs:
            if result.exit_code != 0:
                try:
                    conversation.send_message(f"""
The bounded repair limit is reached. Do not modify any files or attempt a
repair. Update /workspace/openhands-report.json with a diagnosis for attempt
{attempt + 1}, using a failure category and exact evidence quoted from this
captured output, plus root_cause and repair_action set to "no repair attempted;
limit reached". Preserve the actual exit code and output. Do not claim a pass.

Captured output:
{output[-12000:]}
""")
                    conversation.run()
                except Exception as exc:
                    attempts[-1]["diagnosis_error"] = type(exc).__name__
            break
        try:
            conversation.send_message(f"""
The test command failed on attempt {attempt + 1} of {max_repairs + 1}.

Actual output follows:
{output[-12000:]}

Diagnose and classify this failure as dependency, environment, build,
incorrect test, application bug, regression or unknown. Update
/workspace/openhands-report.json with a failures entry containing this attempt
number, category, exact evidence quoted from the output, root_cause and
repair_action. For failed installs, record dependency_installation_failures
with package name, manager, command, exit_code and actual output. Make at most
one minimal sandbox-only repair now. Application fixes must stay in Docker
and have their changes preserved as diffs for developer approval. Update the
report with detected versions/toolchains and all actual verification results.
After repairing, stop; the harness will rerun this same complete-suite test
command and capture its real exit code. Never claim a pass without that run.
""")
            conversation.run()
        except Exception as exc:  # continue once to record the post-repair test state
            attempts[-1]["repair_error"] = type(exc).__name__
    return final_result, attempts

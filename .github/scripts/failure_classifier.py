"""Evidence-based classification of environment and test failures."""

from __future__ import annotations

import re

CATEGORIES = {
    "runtime_missing",
    "interpreter_mismatch",
    "virtualenv_invalid",
    "pip_missing",
    "test_framework_missing",
    "dependency_installation_failed",
    "dependency_conflict",
    "system_library_missing",
    "service_missing",
    "service_startup_failed",
    "database_connection_failed",
    "configuration_missing",
    "test_collection_failed",
    "application_test_failed",
    "timeout",
    "infrastructure_failure",
    "unknown",
}

# Phase 1 repairs only these categories. Everything else is reported as blocked
# instead of attempting an unsafe or unsupported fix.
_PHASE1_REPAIRABLE = {
    "virtualenv_invalid",
    "interpreter_mismatch",
    "pip_missing",
    "test_framework_missing",
    "dependency_installation_failed",
}

_PATTERNS = (
    ("pip_missing", re.compile(r"(?:No module named pip|pip:? command not found|pip is not installed)", re.I)),
    ("test_framework_missing", re.compile(r"(?:No module named pytest|pytest:? command not found|ModuleNotFoundError: No module named 'pytest')", re.I)),
    ("virtualenv_invalid", re.compile(r"(?:pyvenv\.cfg missing|invalid virtual environment|venv path is a symlink|interpreter missing or not executable)", re.I)),
    ("runtime_missing", re.compile(r"(?:python3?: command not found|node: command not found|java: command not found|command not found: python)", re.I)),
    ("dependency_conflict", re.compile(r"(?:ResolutionImpossible|Could not find a version that satisfies|Cannot install .* because|conflicting dependencies|incompatible)", re.I)),
    ("dependency_installation_failed", re.compile(r"(?:No module named|ModuleNotFoundError|ERROR: Could not install|Failed building wheel|Failed to build|metadata-generation-failed)", re.I)),
    ("system_library_missing", re.compile(r"(?:error while loading shared libraries|cannot open shared object file|lib[^\s]+\.so(?:\.\d+)*: cannot open)", re.I)),
    ("database_connection_failed", re.compile(r"(?:authentication failed|database .* does not exist|relation .* does not exist|DB_CONNECTION_ERROR)", re.I)),
    ("service_startup_failed", re.compile(r"(?:container .* (?:exited|unhealthy)|service .* failed to start|health check)", re.I)),
    ("service_missing", re.compile(r"(?:Connection refused|ECONNREFUSED|getaddrinfo ENOTFOUND|Could not connect to (?:postgres|redis|valkey|mysql|mongo))", re.I)),
    ("timeout", re.compile(r"(?:Timed out|TimeoutExpired|timeout after|exceeded .* timeout)", re.I)),
    ("test_collection_failed", re.compile(r"(?:Interrupted: .* errors during collection|ERROR collecting|no tests ran|no tests collected)", re.I)),
    ("application_test_failed", re.compile(r"\bFAILED\b|FAIL:|AssertionError|AssertionFailedError", re.I)),
    ("infrastructure_failure", re.compile(r"(?:credentials|unauthorized|permission denied|docker (?:is not available|daemon)|service unavailable)", re.I)),
)


def _evidence_text(verification: dict) -> str:
    parts: list[str] = []
    for item in verification.get("failures", []) or []:
        parts.append(str(item))
    for check in verification.get("checks", []) or []:
        if not isinstance(check, dict) or check.get("passed") is True:
            continue
        parts.append(f"{check.get('name')}: {check.get('output') or ''}")
    unresolved = verification.get("environment_unresolved")
    if isinstance(unresolved, dict):
        parts.append(str(unresolved))
    return "\n".join(parts)


def _first_match(text: str) -> str | None:
    for category, pattern in _PATTERNS:
        if pattern.search(text):
            return category
    return None


def classify(verification: dict, detection: dict | None = None) -> dict:
    """Return a structured classification with evidence and repairability."""
    text = _evidence_text(verification)
    matched = _first_match(text)
    category = matched or "unknown"
    reason = f"Matched evidence pattern for {category}" if matched else "No specific evidence pattern matched"

    environment = verification.get("python_environment") or {}
    interpreter = verification.get("python_interpreter") or ""
    runner_failed = any(
        isinstance(check, dict)
        and check.get("name") == "python_test_runner"
        and check.get("passed") is False
        for check in verification.get("checks", []) or []
    )
    exit_127 = "127" in text or "No such file or directory" in text

    if verification.get("environment_unresolved"):
        if environment.get("has_pytest") is False:
            category, reason = "test_framework_missing", "Selected project environment lacks pytest"
        elif exit_127:
            category, reason = "virtualenv_invalid", "Environment resolution failed with exit 127"
        else:
            category, reason = "interpreter_mismatch", "No verified interpreter could be resolved"
    elif matched == "test_framework_missing" and environment.get("is_venv") is True:
        reason = "Project virtual environment lacks pytest"
    elif exit_127 and runner_failed and matched != "virtualenv_invalid":
        category, reason = "interpreter_mismatch", "Runner lookup failed under the sanitized environment"
    elif (
        interpreter
        and not interpreter.startswith("/workspace/")
        and category in {"unknown", "test_collection_failed"}
    ):
        category, reason = "interpreter_mismatch", "Selected interpreter is outside the project workspace"

    if category == "test_collection_failed" and "pytest" in text.lower():
        category = "test_framework_missing"

    repair_actions: list[str] = []
    if category in _PHASE1_REPAIRABLE:
        if category in {"virtualenv_invalid", "interpreter_mismatch"}:
            repair_actions = ["ensure_venv", "bootstrap_pip", "install_test_framework", "rebuild_plan"]
        elif category == "pip_missing":
            repair_actions = ["bootstrap_pip", "rebuild_plan"]
        elif category == "test_framework_missing":
            repair_actions = ["install_test_framework"]
            python_detection = (detection or {}).get("python") or {}
            if python_detection.get("requirements_files"):
                repair_actions.append("install_dependencies")
            repair_actions.append("rebuild_plan")
        elif category == "dependency_installation_failed":
            repair_actions = ["install_dependencies", "rebuild_plan"]

    return {
        "category": category if category in CATEGORIES else "unknown",
        "stage": "environment" if category != "application_test_failed" else "tests",
        "reason": reason,
        "evidence": text[-12000:],
        "repairable": category in _PHASE1_REPAIRABLE,
        "repair_actions": repair_actions,
        "blocked_reason": None if category in _PHASE1_REPAIRABLE else f"{category} is not repairable in Phase 1",
    }


def final_status(
    environment_ready: bool,
    suite_exit_code: int | None,
    verification: dict | None = None,
    detection: dict | None = None,
    report_category: str | None = None,
) -> dict:
    """Return truthful overall status and the top-level failure category.

    SUCCESS is only possible when the environment was independently verified
    and the required suite actually executed with exit code 0.
    """
    verification = verification or {}
    history = verification.get("environment_repair_history") or []
    history_categories = [
        item.get("category") for item in history
        if isinstance(item, dict) and item.get("category")
    ]

    if environment_ready and suite_exit_code == 0:
        return {"status": "SUCCESS", "failure_category": None}

    if not environment_ready:
        category = history_categories[-1] if history_categories else classify(
            verification, detection
        )["category"]
        return {"status": "BLOCKED", "failure_category": category}

    category = report_category or history_categories[-1] or "application_test_failed"
    return {"status": "FAILED", "failure_category": category}

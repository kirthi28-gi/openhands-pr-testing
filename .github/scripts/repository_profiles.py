"""Trusted, per-repository automation profiles.

Profiles are read only from the base-branch automation checkout. They are the
only permitted source of service selection and targeted test paths. Agent
output never contributes to a profile.
"""

import json
import re
from pathlib import Path, PurePosixPath


CONFIG_RELATIVE_PATH = Path(".github/config/repository_profiles.json")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
SHELL_UNSAFE_RE = re.compile(r"[;&|<>$\\\s]")


class ProfileError(ValueError):
    """Raised for untrusted or malformed repository profile data."""


def config_path(automation_dir):
    return Path(automation_dir) / CONFIG_RELATIVE_PATH


def safe_test_path(value):
    """Validate a trusted profile path before it can become a pytest argument."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProfileError("Test path must be a non-empty string")
    if SHELL_UNSAFE_RE.search(value) or not SAFE_PATH_RE.fullmatch(value):
        raise ProfileError(f"Unsafe test path: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise ProfileError(f"Unsafe test path: {value}")
    return str(path)


def validate_profile(repository, profile):
    """Return a normalized profile or raise ProfileError."""
    if not isinstance(repository, str) or not REPOSITORY_RE.fullmatch(repository):
        raise ProfileError(f"Invalid repository key: {repository}")
    if not isinstance(profile, dict):
        raise ProfileError(f"Profile must be an object: {repository}")

    normalized = dict(profile)
    targeted = profile.get("targeted_test_paths", {})
    if targeted is None:
        targeted = {}
    if not isinstance(targeted, dict):
        raise ProfileError("targeted_test_paths must be an object")
    checked_targets = {}
    for source, tests in targeted.items():
        source_path = safe_test_path(source)
        if not isinstance(tests, list) or not tests:
            raise ProfileError(f"No test paths configured for {source}")
        checked_targets[source_path] = [safe_test_path(item) for item in tests]
    normalized["targeted_test_paths"] = checked_targets

    always = profile.get("always_test_paths", [])
    if always is None:
        always = []
    if not isinstance(always, list):
        raise ProfileError("always_test_paths must be a list")
    normalized["always_test_paths"] = [safe_test_path(item) for item in always]

    service_profile = profile.get("service_profile")
    if service_profile is not None and (
        not isinstance(service_profile, str)
        or not re.fullmatch(r"[a-z0-9_]+", service_profile)
    ):
        raise ProfileError("service_profile must be a lowercase identifier")
    normalized["service_profile"] = service_profile

    conftest = profile.get("generated_test_conftest")
    if conftest is not None:
        normalized["generated_test_conftest"] = safe_test_path(conftest)

    return normalized


def load_profiles(automation_dir):
    """Load and validate profiles from trusted automation only."""
    path = config_path(automation_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ProfileError(f"Could not read repository profiles: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ProfileError("Profile config must be a version-1 object")
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ProfileError("profiles must be an object")
    return {
        repository: validate_profile(repository, profile)
        for repository, profile in profiles.items()
    }


def profile_for(profiles, repository):
    if not isinstance(repository, str) or not REPOSITORY_RE.fullmatch(repository):
        return None
    profiles = profiles if isinstance(profiles, dict) else {}
    profile = profiles.get(repository)
    return validate_profile(repository, profile) if profile is not None else None


def resolve_test_paths(profile, changed_files):
    """Return trusted test paths for a diff, or None to keep the full suite."""
    profile = validate_profile("trusted/repository", profile) if profile else None
    if profile is None or not profile.get("targeted_test_paths"):
        return None
    files = [
        safe_test_path(item) for item in changed_files
        if isinstance(item, str) and SAFE_PATH_RE.fullmatch(item)
    ]
    selected = set(profile.get("always_test_paths", []))
    for source, tests in profile["targeted_test_paths"].items():
        if any(name == source or name.startswith(source.rstrip("/") + "/") for name in files):
            selected.update(tests)
    return sorted(selected)


def validate_selected_paths(repository_root, paths, allow_missing=()):
    """Fail closed when a selected path is absent or leaves the repository.

    ``tests/generated`` is created inside the sandbox, not in the PR source
    checkout, so trusted profiles may mark it as allowed to be absent here.
    """
    root = Path(repository_root).resolve()
    missing_allowed = {safe_test_path(item) for item in (allow_missing or ())}
    checked = []
    for value in paths:
        path = safe_test_path(value)
        candidate = root / path
        if candidate.is_symlink():
            raise ProfileError(f"Selected test path is a symlink: {path}")
        target = candidate.resolve()
        if not target.is_relative_to(root):
            raise ProfileError(f"Selected test path escapes the repository: {path}")
        if not target.exists() and path not in missing_allowed:
            raise ProfileError(f"Selected test path does not exist: {path}")
        checked.append(path)
    return checked


def build_targeted_test_command(profile, changed_files, repository_root, interpreter):
    """Build a pytest command exclusively from trusted profile data."""
    paths = resolve_test_paths(profile, changed_files)
    if paths is None:
        return None, set(), set()
    checked = validate_selected_paths(
        repository_root, paths, allow_missing=("tests/generated",)
    )
    if not checked:
        raise ProfileError("Targeted profile selected no test paths")
    import shlex
    command = shlex.join([interpreter, "-m", "pytest", *checked])
    return command, set(checked), set(checked)

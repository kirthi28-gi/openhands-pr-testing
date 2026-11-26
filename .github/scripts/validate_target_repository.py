#!/usr/bin/env python3
"""Validate controlled workflow inputs before any PR source is checked out."""

import argparse
import json
import os
import re
import sys
from pathlib import Path


REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def validate_inputs(repository, base_sha, head_sha, pr_number, automation_dir):
    """Fail closed for unapproved repositories or malformed PR identifiers."""
    if not isinstance(repository, str) or not REPOSITORY_RE.fullmatch(repository):
        raise ValueError("Invalid target repository")
    config_path = Path(automation_dir) / ".github/config/repository_profiles.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Could not read trusted repository profiles: {exc}") from exc
    profiles = config.get("profiles", {}) if isinstance(config, dict) else {}
    if not isinstance(profiles, dict):
        raise ValueError("Malformed repository profiles")
    allowed = set(profiles) | {os.getenv("GITHUB_REPOSITORY", "")}
    if repository not in allowed:
        raise ValueError(f"Target repository is not approved: {repository}")

    for name, value in (("base_sha", base_sha), ("head_sha", head_sha)):
        if not isinstance(value, str) or not SHA_RE.fullmatch(value):
            raise ValueError(f"Invalid {name}")
    try:
        number = int(pr_number)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid PR number") from exc
    if number <= 0:
        raise ValueError("Invalid PR number")
    return {"repository": repository, "pr_number": number}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automation-dir", required=True)
    args = parser.parse_args()
    try:
        validate_inputs(
            os.getenv("TARGET_REPOSITORY", ""),
            os.getenv("PR_BASE_SHA", ""),
            os.getenv("PR_HEAD_SHA", ""),
            os.getenv("PR_NUMBER", ""),
            args.automation_dir,
        )
    except ValueError as exc:
        print(f"Workflow input validation failed: {exc}", file=sys.stderr)
        return 1
    print("Workflow inputs validated against trusted configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

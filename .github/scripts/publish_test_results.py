#!/usr/bin/env python3
"""Validate OpenHands findings against a PR diff and publish GitHub comments."""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath


INLINE_CATEGORIES = {"application bug", "regression"}
SUMMARY_MARKER = "<!-- openhands-test-generation-summary -->"
INLINE_MARKER_PREFIX = "openhands-test-generation-inline"
DEFAULT_GITHUB_API_URL = "https://api.github.com"
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


class PublisherError(RuntimeError):
    """Raised when publication cannot be performed safely."""


class GitHubAPIError(RuntimeError):
    """Raised when a GitHub API request fails."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def safe_repo_path(value):
    """Return a safe repository-relative POSIX path, or None."""
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    if "\\" in value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        return None

    path = PurePosixPath(value)
    if path.is_absolute() or str(path) in {"", "."}:
        return None
    if ".." in path.parts:
        return None
    return value


def parse_added_lines(diff_text):
    """Map each changed file to the line numbers added on the RIGHT side."""
    added = {}
    current_file = None
    new_line = None
    in_hunk = False

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            raw_path = line[4:]
            if raw_path == "/dev/null":
                current_file = None
            elif raw_path.startswith("b/") and not raw_path.startswith('b/"'):
                current_file = safe_repo_path(raw_path[2:])
                if current_file:
                    added.setdefault(current_file, set())
            else:
                # Fail closed for quoted or unexpected Git paths.
                current_file = None
            new_line = None
            in_hunk = False
            continue

        match = HUNK_RE.match(line)
        if match:
            new_line = int(match.group(3))
            in_hunk = current_file is not None
            continue

        if not in_hunk or new_line is None:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            added[current_file].add(new_line)
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            # A deletion does not advance the new-file line number.
            continue
        elif line.startswith("\\"):
            continue
        else:
            # Context line.
            new_line += 1

    return added


def validate_inline_candidate(failure, added_lines):
    """Return a verified inline location, or a summary-only reason."""
    if not isinstance(failure, dict):
        return None, "Invalid failure record"

    category = failure.get("category")
    if category not in INLINE_CATEGORIES:
        return None, f"Category '{category}' is summary-only"

    evidence = failure.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(
            not isinstance(item, str) or not item.strip() for item in evidence
        )
    ):
        return None, "Failure has no validated evidence"

    root_cause = failure.get("root_cause")
    if not isinstance(root_cause, str) or not root_cause.strip():
        return None, "Failure has no validated root cause"

    filename = safe_repo_path(failure.get("affected_file"))
    if filename is None:
        return None, "No safe source-file candidate was supplied"

    line = failure.get("affected_line")
    if not isinstance(line, int) or isinstance(line, bool) or line <= 0:
        return None, "No valid positive source line was supplied"

    if filename not in added_lines:
        return None, "Candidate file is not present as an added file in the PR diff"

    if line not in added_lines[filename]:
        return None, "Candidate line is not an added line in the PR diff"

    return {"path": filename, "line": line}, None


def format_inline_comment(failure):
    """Build an evidence-based inline comment from validated report fields."""
    failing_test = failure.get("failing_test") or "Failing test not specified"
    root_cause = failure.get("root_cause") or "Root cause not specified"
    repair_action = failure.get("repair_action") or "No repair suggestion was recorded"
    evidence = failure.get("evidence")
    if not isinstance(evidence, list):
        evidence = []

    evidence_text = "\n".join(f"> {item}" for item in evidence if isinstance(item, str))
    if not evidence_text:
        evidence_text = "> No validated evidence was recorded."

    return (
        f"**OpenHands test finding — {failure.get('category', 'unknown')}**\n\n"
        f"**Failing test:** `{failing_test}`\n\n"
        f"**Why it failed:** {root_cause}\n\n"
        f"**Evidence:**\n{evidence_text}\n\n"
        f"**Suggested action:** {repair_action}"
    )


def build_summary(report, workflow_result="unknown"):
    """Create a readable summary without inventing missing report details."""
    if not isinstance(report, dict):
        report = {}

    failures = report.get("failures")
    if not isinstance(failures, list):
        failures = []

    lines = [
        SUMMARY_MARKER,
        "## OpenHands PR test-generation results",
        "",
        f"- **Workflow result:** {workflow_result}",
        f"- **Report result:** {report.get('result', 'Not available')}",
        f"- **Test execution result:** {report.get('harness_result', report.get('result', 'Not available'))}",
        f"- **Failure category:** {report.get('final_failure_category', 'Not classified')}",
        f"- **Failures with validated diagnoses:** {len(failures)}",
        "",
        "### Findings",
        "",
    ]

    if not failures:
        lines.append("No validated failure diagnoses were included in the report.")
    else:
        for index, failure in enumerate(failures, start=1):
            if not isinstance(failure, dict):
                continue
            category = failure.get("category", "unknown")
            test = failure.get("failing_test", "Test not specified")
            cause = failure.get("root_cause", "Root cause not available")
            lines.extend([
                f"**{index}. {test}** — `{category}`",
                "",
                str(cause),
                "",
            ])
            evidence = failure.get("evidence")
            if isinstance(evidence, list) and evidence:
                lines.append("Evidence:")
                for item in evidence:
                    if isinstance(item, str):
                        lines.append(f"- `{item}`")
                lines.append("")
            repair_action = failure.get("repair_action")
            if isinstance(repair_action, str) and repair_action.strip():
                lines.extend(["Suggested repair:", "", repair_action, ""])

    return "\n".join(lines).strip()


def report_passes_validation(report):
    """Independently check artifact validation state, not just agent flags.

    The artifact is untrusted input. A validation flag or an empty error list
    supplied by the agent is not evidence that validation happened, so both the
    computed flags and their underlying error lists must be clean.
    """
    if not isinstance(report, dict):
        return False
    errors = report.get("report_validation_errors")
    if not isinstance(errors, list) or errors:
        return False
    diff_errors = report.get("application_diff_errors")
    if not isinstance(diff_errors, list) or diff_errors:
        return False
    if report.get("application_diff_valid") is not True:
        return False
    agent_valid = report.get("agent_report_valid")
    if agent_valid is None:
        agent_valid = report.get("report_valid")
    if agent_valid is not True:
        return False
    return report.get("report_valid", agent_valid) is True


def trusted_diagnoses(report):
    """Return harness-validated diagnoses keyed by attempt from an artifact."""
    if not isinstance(report, dict):
        return {}
    attempts = report.get("harness_attempts")
    if not isinstance(attempts, list):
        return {}

    validated = {}
    for item in attempts:
        if not isinstance(item, dict):
            continue
        attempt = item.get("attempt")
        diagnosis = item.get("diagnosis")
        if (
            isinstance(attempt, int)
            and not isinstance(attempt, bool)
            and isinstance(diagnosis, dict)
            and diagnosis.get("validated") is True
        ):
            validated[attempt] = diagnosis
    return validated


def matches_trusted_diagnosis(failure, diagnosis):
    """Return None when a finding exactly matches a harness-validated diagnosis."""
    if not isinstance(diagnosis, dict) or diagnosis.get("validated") is not True:
        return "Diagnosis was not validated by the trusted harness"
    if diagnosis.get("category") not in INLINE_CATEGORIES:
        return "Trusted diagnosis is not a validated application bug or regression"
    for key in (
        "category",
        "evidence",
        "root_cause",
        "failing_test",
        "root_cause_evidence",
        "repair_action",
    ):
        if failure.get(key) != diagnosis.get(key):
            return f"Finding {key} does not match the trusted diagnosis"
    for key in ("affected_file", "affected_line"):
        if failure.get(key) != diagnosis.get(key):
            return f"Finding {key} is not the validated source location"
    return None


def inspect_report_data(report, diff_text, workflow_result="unknown"):
    """Validate report findings against already-trusted diff text."""
    if not isinstance(report, dict):
        raise TypeError("Report JSON must contain an object at the top level")

    added_lines = parse_added_lines(diff_text)

    failures = report.get("failures", [])
    if not isinstance(failures, list):
        failures = []

    report_valid = report_passes_validation(report)
    validated_diagnoses = trusted_diagnoses(report)
    inline = []
    summary_only = []

    for failure in failures:
        if report_valid:
            location, reason = validate_inline_candidate(failure, added_lines)
            if location:
                attempt = failure.get("attempt")
                diagnosis = (
                    validated_diagnoses.get(attempt)
                    if isinstance(attempt, int) and not isinstance(attempt, bool)
                    else None
                )
                mismatch = matches_trusted_diagnosis(failure, diagnosis)
                if mismatch:
                    location, reason = None, mismatch
        else:
            location, reason = None, "Report did not pass trusted validation"
        if location:
            inline.append({
                "location": location,
                "comment": format_inline_comment(failure),
            })
        else:
            summary_only.append({
                "failure": failure,
                "reason": reason,
            })

    return {
        "inline": inline,
        "summary_only": summary_only,
        "summary": build_summary(report, workflow_result),
    }


def inspect_report(report_path, diff_path, workflow_result="unknown"):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    diff_text = Path(diff_path).read_text(encoding="utf-8")
    return inspect_report_data(report, diff_text, workflow_result)


def inline_marker(pr_number, base_sha, head_sha, path, line):
    """Return a stable identifier for one inline finding on one PR revision."""
    key = f"{pr_number}|{base_sha}|{head_sha}|{path}|{line}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"<!-- {INLINE_MARKER_PREFIX} {digest} -->"


def validate_publish_context(repository, pr_number, base_sha, head_sha):
    """Validate the trusted workflow context used for GitHub API calls."""
    if not isinstance(repository, str) or not REPOSITORY_RE.fullmatch(repository):
        raise PublisherError("Invalid GitHub repository context")
    if (
        not isinstance(pr_number, int)
        or isinstance(pr_number, bool)
        or pr_number <= 0
    ):
        raise PublisherError("Invalid PR number")
    for name, value in (("base_sha", base_sha), ("head_sha", head_sha)):
        if not isinstance(value, str) or not SHA_RE.fullmatch(value):
            raise PublisherError(f"Invalid {name}")


def validate_report_context(report, repository, pr_number, base_sha, head_sha):
    """Refuse to publish an artifact produced for a different PR revision."""
    reported_repository = report.get("target_repository")
    if (
        isinstance(reported_repository, str)
        and reported_repository
        and reported_repository.lower() != repository.lower()
    ):
        raise PublisherError("Report target repository does not match workflow context")
    reported_pr = report.get("pr_number")
    if reported_pr is not None and (
        not isinstance(reported_pr, int)
        or isinstance(reported_pr, bool)
        or reported_pr != pr_number
    ):
        raise PublisherError("Report PR number does not match workflow context")
    for key, expected in (("base_sha", base_sha), ("head_sha", head_sha)):
        value = report.get(key)
        if isinstance(value, str) and value and value.lower() != expected.lower():
            raise PublisherError(f"Report {key} does not match workflow context")


def github_api_request(method, path, payload=None, token=None, api_url=DEFAULT_GITHUB_API_URL):
    """Make one GitHub REST API request using only stdlib networking."""
    if not token:
        raise PublisherError("A GitHub token is required for publishing")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "openhands-regression-pilot",
    }
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{api_url}{path}", data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise GitHubAPIError(
            f"GitHub API {method} {path} failed with HTTP {exc.code}: {detail}",
            status=exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        raise GitHubAPIError(
            f"GitHub API {method} {path} failed: {exc.reason}"
        ) from exc

    return json.loads(content) if content.strip() else {}


def list_comments(path, token, api_request, page_limit=10):
    """List comments with bounded pagination for duplicate detection."""
    items = []
    for page in range(1, page_limit + 1):
        separator = "&" if "?" in path else "?"
        batch = api_request(
            "GET", f"{path}{separator}per_page=100&page={page}", token=token
        )
        if not isinstance(batch, list):
            break
        items.extend(batch)
        if len(batch) < 100:
            break
    return items


def publish_results(
    report,
    diff_text,
    workflow_result,
    repository,
    pr_number,
    base_sha,
    head_sha,
    token,
    api_request=None,
):
    """Publish validated inline findings and one updatable PR summary."""
    if not token:
        raise PublisherError("A GitHub token is required for publishing")
    api_request = api_request or github_api_request

    validate_publish_context(repository, pr_number, base_sha, head_sha)
    if not isinstance(report, dict):
        raise TypeError("Report JSON must contain an object at the top level")
    validate_report_context(report, repository, pr_number, base_sha, head_sha)

    inspection = inspect_report_data(report, diff_text, workflow_result)

    review_path = f"/repos/{repository}/pulls/{pr_number}/comments"
    issue_path = f"/repos/{repository}/issues/{pr_number}/comments"
    existing_review_comments = list_comments(review_path, token, api_request)
    existing_issue_comments = list_comments(issue_path, token, api_request)

    inline_created = []
    inline_skipped = []
    for item in inspection["inline"]:
        location = item["location"]
        path = location["path"]
        line = location["line"]
        marker = inline_marker(pr_number, base_sha, head_sha, path, line)
        if any(
            isinstance(comment, dict)
            and marker in str(comment.get("body") or "")
            for comment in existing_review_comments
        ):
            inline_skipped.append({"path": path, "line": line})
            continue

        api_request(
            "POST",
            review_path,
            payload={
                "path": path,
                "line": line,
                "side": "RIGHT",
                "commit_id": head_sha,
                "body": f"{marker}\n\n{item['comment']}",
            },
            token=token,
        )
        inline_created.append({"path": path, "line": line})

    summary = inspection["summary"]
    existing_summary = next(
        (
            comment
            for comment in existing_issue_comments
            if isinstance(comment, dict)
            and SUMMARY_MARKER in str(comment.get("body") or "")
            and isinstance(comment.get("id"), int)
        ),
        None,
    )
    if existing_summary is None:
        api_request("POST", issue_path, payload={"body": summary}, token=token)
        summary_created, summary_updated = True, False
    else:
        api_request(
            "PATCH",
            f"/repos/{repository}/issues/comments/{existing_summary['id']}",
            payload={"body": summary},
            token=token,
        )
        summary_created, summary_updated = False, True

    return {
        "inline_created": inline_created,
        "inline_skipped": inline_skipped,
        "summary_created": summary_created,
        "summary_updated": summary_updated,
        "summary_only": len(inspection["summary_only"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Path to openhands-report.json")
    parser.add_argument("--diff", required=True, help="Path to the actual PR unified diff")
    parser.add_argument("--workflow-result", default="unknown")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish validated findings to GitHub (requires a token).",
    )
    parser.add_argument("--repository", help="owner/repo of the pull request")
    parser.add_argument("--pr-number", type=int)
    parser.add_argument("--base-sha")
    parser.add_argument("--head-sha")
    args = parser.parse_args()

    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        diff_text = Path(args.diff).read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"Publisher inspection failed: {exc}", file=sys.stderr)
        return 1

    if args.publish:
        try:
            outcome = publish_results(
                report=report,
                diff_text=diff_text,
                workflow_result=args.workflow_result,
                repository=args.repository,
                pr_number=args.pr_number,
                base_sha=args.base_sha,
                head_sha=args.head_sha,
                token=os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"),
            )
        except (PublisherError, GitHubAPIError, TypeError, ValueError) as exc:
            print(f"Publishing failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(outcome, indent=2))
        return 0

    try:
        result = inspect_report_data(report, diff_text, args.workflow_result)
    except (TypeError, ValueError) as exc:
        print(f"Publisher inspection failed: {exc}", file=sys.stderr)
        return 1

    print("=== INLINE COMMENT CANDIDATES ===")
    if not result["inline"]:
        print("None")
    for item in result["inline"]:
        location = item["location"]
        print(f"\n{location['path']}:{location['line']}")
        print(item["comment"])

    print("\n=== SUMMARY-ONLY FINDINGS ===")
    if not result["summary_only"]:
        print("None")
    for item in result["summary_only"]:
        failure = item["failure"]
        test = failure.get("failing_test", "Test not specified") if isinstance(failure, dict) else "Invalid failure"
        print(f"- {test}: {item['reason']}")

    print("\n=== PR SUMMARY PREVIEW ===")
    print(result["summary"])
    print(
        f"\nInspection complete: {len(result['inline'])} inline candidate(s), "
        f"{len(result['summary_only'])} summary-only finding(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

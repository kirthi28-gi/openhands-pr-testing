import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / ".github/scripts/publish_test_results.py"
SPEC = importlib.util.spec_from_file_location("publish_test_results_github", SCRIPT)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


REPOSITORY = "acme/widgets"
PR_NUMBER = 42
BASE_SHA = "1" * 40
HEAD_SHA = "2" * 40

DIFF = """\
diff --git a/app/calculator.py b/app/calculator.py
index 1111111..2222222 100644
--- a/app/calculator.py
+++ b/app/calculator.py
@@ -10,2 +10,4 @@
 existing_line()
+def calculate_total():
+    return 42
 another_existing_line()
"""


def sample_report(**overrides):
    report = {
        "pr_number": PR_NUMBER,
        "base_sha": BASE_SHA,
        "head_sha": HEAD_SHA,
        "result": "failed",
        "harness_result": "failed",
        "final_failure_category": "application bug",
        "agent_report_valid": True,
        "report_valid": True,
        "report_validation_errors": [],
        "application_diff_valid": True,
        "application_diff_errors": [],
        "failures": [
            {
                "attempt": 1,
                "category": "application bug",
                "failing_test": "tests/test_calculator.py::test_total",
                "affected_file": "app/calculator.py",
                "affected_line": 11,
                "root_cause": "The total calculation returns an incorrect value.",
                "evidence": ["AssertionError: expected 42, got 41"],
                "root_cause_evidence": ["AssertionError: expected 42, got 41"],
                "repair_action": "Correct the calculation.",
            }
        ],
        "harness_attempts": [
            {
                "attempt": 1,
                "diagnosis": {
                    "attempt": 1,
                    "validated": True,
                    "category": "application bug",
                    "failing_test": "tests/test_calculator.py::test_total",
                    "affected_file": "app/calculator.py",
                    "affected_line": 11,
                    "root_cause": "The total calculation returns an incorrect value.",
                    "root_cause_evidence": ["AssertionError: expected 42, got 41"],
                    "evidence": ["AssertionError: expected 42, got 41"],
                    "repair_action": "Correct the calculation.",
                },
            }
        ],
    }
    report.update(overrides)
    return report


class FakeGitHub:
    """Records API calls and serves canned comment lists without networking."""

    def __init__(self, review_comments=None, issue_comments=None, failure=None):
        self.calls = []
        self.review_comments = review_comments or []
        self.issue_comments = issue_comments or []
        self.failure = failure

    def __call__(self, method, path, payload=None, token=None, api_url=None):
        self.calls.append(
            {
                "method": method,
                "path": path,
                "payload": payload,
                "token": token,
            }
        )
        if self.failure and self.failure[0] == method and self.failure[1] in path:
            raise publisher.GitHubAPIError("HTTP 403: forbidden", status=403)
        if method == "GET" and "/pulls/" in path:
            return self.review_comments
        if method == "GET" and "/issues/" in path:
            return self.issue_comments
        return {"id": 12345}


def matching(fake, method, fragment):
    return [
        call
        for call in fake.calls
        if call["method"] == method and fragment in call["path"]
    ]


def publish(report=None, fake=None, token="test-token", **kwargs):
    api = fake if fake is not None else FakeGitHub()
    options = {
        "report": report if report is not None else sample_report(),
        "diff_text": DIFF,
        "workflow_result": "failure",
        "repository": REPOSITORY,
        "pr_number": PR_NUMBER,
        "base_sha": BASE_SHA,
        "head_sha": HEAD_SHA,
        "token": token,
        "api_request": api,
    }
    options.update(kwargs)
    outcome = publisher.publish_results(**options)
    return outcome, api


def test_publishes_inline_comment_for_validated_application_bug():
    outcome, api = publish()

    comments = matching(api, "POST", "/pulls/42/comments")
    assert len(comments) == 1
    payload = comments[0]["payload"]
    assert payload["path"] == "app/calculator.py"
    assert payload["line"] == 11
    assert payload["side"] == "RIGHT"
    assert payload["commit_id"] == HEAD_SHA
    assert "test_total" in payload["body"]
    assert "expected 42, got 41" in payload["body"]
    assert publisher.INLINE_MARKER_PREFIX in payload["body"]
    assert outcome["inline_created"] == [{"path": "app/calculator.py", "line": 11}]
    assert outcome["inline_skipped"] == []
    assert matching(api, "POST", "/issues/42/comments")
    assert not matching(api, "PATCH", "/issues/comments/")


def test_regression_diagnosis_is_eligible_for_inline_comment():
    report = sample_report()
    report["final_failure_category"] = "regression"
    report["failures"][0]["category"] = "regression"
    report["harness_attempts"][0]["diagnosis"]["category"] = "regression"
    outcome, api = publish(report=report)
    assert len(matching(api, "POST", "/pulls/42/comments")) == 1
    assert outcome["inline_created"]


def test_duplicate_inline_comment_is_skipped():
    marker = publisher.inline_marker(
        PR_NUMBER, BASE_SHA, HEAD_SHA, "app/calculator.py", 11
    )
    api = FakeGitHub(review_comments=[{"id": 7, "body": f"{marker}\nold finding"}])
    outcome, _ = publish(fake=api)

    assert outcome["inline_created"] == []
    assert outcome["inline_skipped"] == [{"path": "app/calculator.py", "line": 11}]
    assert not matching(api, "POST", "/pulls/42/comments")


def test_existing_summary_is_updated_instead_of_duplicated():
    api = FakeGitHub(
        issue_comments=[
            {"id": 555, "body": "notes"},
            {"id": 777, "body": f"old\n{publisher.SUMMARY_MARKER}"},
        ]
    )
    outcome, api = publish(fake=api)

    updates = matching(api, "PATCH", "/issues/comments/777")
    assert len(updates) == 1
    assert publisher.SUMMARY_MARKER in updates[0]["payload"]["body"]
    assert not matching(api, "POST", "/issues/42/comments")
    assert outcome["summary_updated"] is True
    assert outcome["summary_created"] is False


def test_summary_only_failures_do_not_create_inline_comments():
    report = sample_report(final_failure_category="dependency")
    report["failures"][0].update(
        category="dependency", affected_file=None, affected_line=None
    )
    outcome, api = publish(report=report)

    assert not matching(api, "POST", "/pulls/42/comments")
    assert matching(api, "POST", "/issues/42/comments")
    assert outcome["inline_created"] == []
    assert outcome["summary_only"] == 1


def test_incorrect_test_diagnosis_is_summary_only():
    report = sample_report()
    report["failures"][0]["category"] = "incorrect test"
    outcome, api = publish(report=report)
    assert not matching(api, "POST", "/pulls/42/comments")
    assert outcome["summary_only"] == 1


def test_line_not_added_in_diff_is_summary_only():
    report = sample_report()
    report["failures"][0]["affected_line"] = 10
    outcome, api = publish(report=report)
    assert not matching(api, "POST", "/pulls/42/comments")
    assert outcome["summary_only"] == 1


def test_unsafe_location_is_summary_only():
    report = sample_report()
    report["failures"][0]["affected_file"] = "../../etc/passwd"
    outcome, api = publish(report=report)
    assert not matching(api, "POST", "/pulls/42/comments")
    assert outcome["summary_only"] == 1


def test_unvalidated_report_produces_no_inline_comments():
    report = sample_report(agent_report_valid=False, report_validation_errors=["bad"])
    outcome, api = publish(report=report)
    assert not matching(api, "POST", "/pulls/42/comments")
    assert matching(api, "POST", "/issues/42/comments")
    assert outcome["summary_only"] == 1


def test_missing_token_is_rejected_before_any_api_call():
    api = FakeGitHub()
    with pytest.raises(publisher.PublisherError, match="token"):
        publish(fake=api, token=None)
    assert api.calls == []


def test_missing_permissions_error_is_surfaced():
    api = FakeGitHub(failure=("GET", "/issues/42/comments"))
    with pytest.raises(publisher.GitHubAPIError) as exc_info:
        publish(fake=api)
    assert exc_info.value.status == 403
    assert not matching(api, "POST", "/pulls/42/comments")


def test_inline_comment_api_error_is_surfaced():
    api = FakeGitHub(failure=("POST", "/pulls/42/comments"))
    with pytest.raises(publisher.GitHubAPIError) as exc_info:
        publish(fake=api)
    assert exc_info.value.status == 403
    assert not matching(api, "POST", "/issues/42/comments")


def test_invalid_report_is_rejected_before_publishing():
    api = FakeGitHub()
    with pytest.raises(TypeError, match="top level"):
        publish(report=[], fake=api)
    assert api.calls == []


def test_report_context_mismatch_is_rejected():
    api = FakeGitHub()
    report = sample_report(pr_number=999)
    with pytest.raises(publisher.PublisherError, match="PR number"):
        publish(report=report, fake=api)
    assert api.calls == []


def test_invalid_workflow_context_is_rejected():
    api = FakeGitHub()
    with pytest.raises(publisher.PublisherError):
        publish(fake=api, head_sha="not-a-sha")
    assert api.calls == []


def test_inline_marker_is_stable_for_one_revision():
    first = publisher.inline_marker(
        PR_NUMBER, BASE_SHA, HEAD_SHA, "app/calculator.py", 11
    )
    assert first == publisher.inline_marker(
        PR_NUMBER, BASE_SHA, HEAD_SHA, "app/calculator.py", 11
    )
    assert first != publisher.inline_marker(
        PR_NUMBER, BASE_SHA, "3" * 40, "app/calculator.py", 11
    )


def test_summary_includes_repair_information_and_marker():
    summary = publisher.build_summary(sample_report(), workflow_result="failure")
    assert publisher.SUMMARY_MARKER in summary
    assert "Correct the calculation." in summary
    assert "expected 42, got 41" in summary


def test_agent_valid_flag_cannot_override_trusted_validation_errors():
    report = sample_report(
        agent_report_valid=True,
        report_valid=True,
        report_validation_errors=["diagnosis evidence does not match captured output"],
    )
    outcome, api = publish(report=report)
    assert not matching(api, "POST", "/pulls/42/comments")
    assert outcome["inline_created"] == []
    assert outcome["summary_only"] == 1


def test_agent_supplied_report_valid_flag_is_not_trusted_alone():
    report = sample_report(agent_report_valid=False, report_valid=True)
    outcome, api = publish(report=report)
    assert not matching(api, "POST", "/pulls/42/comments")
    assert outcome["inline_created"] == []


def test_manipulated_empty_validation_errors_cannot_hide_fabricated_evidence():
    report = sample_report(report_validation_errors=[])
    report["failures"][0]["evidence"] = ["E AssertionError: expected 999, got 0"]
    outcome, api = publish(report=report)
    assert not matching(api, "POST", "/pulls/42/comments")
    assert outcome["summary_only"] == 1


def test_missing_trusted_validation_fields_are_summary_only():
    report = sample_report()
    del report["report_validation_errors"]
    outcome, api = publish(report=report)
    assert not matching(api, "POST", "/pulls/42/comments")
    assert outcome["summary_only"] == 1


def test_invalid_diff_capture_blocks_inline_comments():
    report = sample_report(
        agent_report_valid=True,
        application_diff_valid=False,
        application_diff_errors=["Sandbox application diff could not be captured as a valid patch."],
    )
    outcome, api = publish(report=report)
    assert not matching(api, "POST", "/pulls/42/comments")
    assert outcome["inline_created"] == []
    assert matching(api, "POST", "/issues/42/comments")


def test_unvalidated_diagnosis_is_summary_only():
    report = sample_report()
    report["harness_attempts"][0]["diagnosis"]["validated"] = False
    outcome, api = publish(report=report)
    assert not matching(api, "POST", "/pulls/42/comments")
    assert outcome["summary_only"] == 1


def test_finding_without_trusted_diagnosis_is_summary_only():
    report = sample_report()
    report["harness_attempts"] = []
    outcome, api = publish(report=report)
    assert not matching(api, "POST", "/pulls/42/comments")
    assert outcome["summary_only"] == 1


def test_finding_with_invalid_attempt_is_summary_only():
    report = sample_report()
    report["failures"][0]["attempt"] = "1"
    outcome, api = publish(report=report)
    assert not matching(api, "POST", "/pulls/42/comments")
    assert outcome["summary_only"] == 1


def test_report_passes_validation_rejects_agent_flags_without_validation_state():
    report = sample_report()
    del report["application_diff_valid"]
    assert publisher.report_passes_validation(report) is False
    assert publisher.report_passes_validation([]) is False
    assert publisher.report_passes_validation({"agent_report_valid": True}) is False


def test_trusted_diagnoses_ignores_unvalidated_records():
    report = sample_report()
    report["harness_attempts"][0]["diagnosis"]["validated"] = False
    assert publisher.trusted_diagnoses(report) == {}
    assert 1 in publisher.trusted_diagnoses(sample_report())


def test_cli_publish_requires_token(tmp_path, monkeypatch, capsys):
    report = tmp_path / "report.json"
    diff = tmp_path / "changes.diff"
    report.write_text(json.dumps(sample_report()), encoding="utf-8")
    diff.write_text(DIFF, encoding="utf-8")

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "publish_test_results.py",
            "--publish",
            "--report",
            str(report),
            "--diff",
            str(diff),
            "--repository",
            REPOSITORY,
            "--pr-number",
            str(PR_NUMBER),
            "--base-sha",
            BASE_SHA,
            "--head-sha",
            HEAD_SHA,
        ],
    )
    assert publisher.main() == 1
    assert "token" in capsys.readouterr().err

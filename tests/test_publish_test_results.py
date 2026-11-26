import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / ".github/scripts/publish_test_results.py"
SPEC = importlib.util.spec_from_file_location("publish_test_results", SCRIPT)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


SAMPLE_DIFF = """\
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


def sample_failure(**overrides):
    failure = {
        "category": "application bug",
        "failing_test": "tests/test_calculator.py::test_total",
        "affected_file": "app/calculator.py",
        "affected_line": 11,
        "root_cause": "The total calculation returns an incorrect value.",
        "evidence": ["AssertionError: expected 42, got 41"],
        "repair_action": "Correct the calculation.",
    }
    failure.update(overrides)
    return failure


def test_parser_finds_added_lines():
    added = publisher.parse_added_lines(SAMPLE_DIFF)
    assert added == {"app/calculator.py": {11, 12}}


def test_valid_application_bug_can_be_inline():
    location, reason = publisher.validate_inline_candidate(
        sample_failure(), publisher.parse_added_lines(SAMPLE_DIFF)
    )
    assert location == {"path": "app/calculator.py", "line": 11}
    assert reason is None


def test_valid_regression_can_be_inline():
    failure = sample_failure(category="regression")
    location, reason = publisher.validate_inline_candidate(
        failure, publisher.parse_added_lines(SAMPLE_DIFF)
    )
    assert location == {"path": "app/calculator.py", "line": 11}
    assert reason is None


def test_dependency_failure_is_summary_only():
    failure = sample_failure(category="dependency")
    location, reason = publisher.validate_inline_candidate(
        failure, publisher.parse_added_lines(SAMPLE_DIFF)
    )
    assert location is None
    assert "summary-only" in reason


def test_incorrect_test_is_summary_only():
    failure = sample_failure(category="incorrect test")
    location, reason = publisher.validate_inline_candidate(
        failure, publisher.parse_added_lines(SAMPLE_DIFF)
    )
    assert location is None
    assert "summary-only" in reason


def test_line_not_added_in_diff_is_summary_only():
    failure = sample_failure(affected_line=10)
    location, reason = publisher.validate_inline_candidate(
        failure, publisher.parse_added_lines(SAMPLE_DIFF)
    )
    assert location is None
    assert "not an added line" in reason


def test_file_not_in_diff_is_summary_only():
    failure = sample_failure(affected_file="app/other.py")
    location, reason = publisher.validate_inline_candidate(
        failure, publisher.parse_added_lines(SAMPLE_DIFF)
    )
    assert location is None
    assert "not present" in reason


def test_path_traversal_is_rejected():
    failure = sample_failure(affected_file="../outside.py")
    location, reason = publisher.validate_inline_candidate(
        failure, publisher.parse_added_lines(SAMPLE_DIFF)
    )
    assert location is None
    assert "safe source-file" in reason


def test_absolute_path_is_rejected():
    failure = sample_failure(affected_file="/etc/passwd")
    location, reason = publisher.validate_inline_candidate(
        failure, publisher.parse_added_lines(SAMPLE_DIFF)
    )
    assert location is None


def test_boolean_line_number_is_rejected():
    failure = sample_failure(affected_line=True)
    location, reason = publisher.validate_inline_candidate(
        failure, publisher.parse_added_lines(SAMPLE_DIFF)
    )
    assert location is None
    assert "positive source line" in reason


def test_zero_line_number_is_rejected():
    failure = sample_failure(affected_line=0)
    location, reason = publisher.validate_inline_candidate(
        failure, publisher.parse_added_lines(SAMPLE_DIFF)
    )
    assert location is None


def test_missing_location_is_summary_only():
    failure = sample_failure(affected_file=None, affected_line=None)
    location, reason = publisher.validate_inline_candidate(
        failure, publisher.parse_added_lines(SAMPLE_DIFF)
    )
    assert location is None


def test_inline_comment_contains_diagnosis_and_evidence():
    comment = publisher.format_inline_comment(sample_failure())
    assert "test_total" in comment
    assert "incorrect value" in comment
    assert "AssertionError: expected 42, got 41" in comment
    assert "Correct the calculation." in comment


def test_summary_has_update_marker_and_failure_details():
    summary = publisher.build_summary(
        {"result": "failure", "failures": [sample_failure()]},
        workflow_result="failure",
    )
    assert "<!-- openhands-test-generation-summary -->" in summary
    assert "test_total" in summary
    assert "application bug" in summary


def test_empty_report_summary_is_safe():
    summary = publisher.build_summary({}, workflow_result="failure")
    assert "No validated failure diagnoses" in summary


def test_invalid_failure_record_is_summary_only():
    location, reason = publisher.validate_inline_candidate(
        None, publisher.parse_added_lines(SAMPLE_DIFF)
    )
    assert location is None
    assert reason == "Invalid failure record"


def test_inspect_report_rejects_non_object_json(tmp_path):
    report = tmp_path / "report.json"
    diff = tmp_path / "changes.diff"

    report.write_text("[]", encoding="utf-8")
    diff.write_text(SAMPLE_DIFF, encoding="utf-8")

    import pytest

    with pytest.raises(TypeError, match="top level"):
        publisher.inspect_report(report, diff)


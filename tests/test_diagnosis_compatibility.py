"""Focused tests for Fix 2 diagnosis report compatibility."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

import generate_tests as gt  # noqa: E402
from pr_test_harness import load_agent_report  # noqa: E402

OUTPUT = "FAILED tests/test_math.py::test_add\nE AssertionError: expected 3, got 2"
ATTEMPTS = [{"attempt": 1, "exit_code": 1, "output": OUTPUT}]


def _report(category="application_bug", evidence=None, root_cause_evidence=None):
    return json.dumps({
        "schema_version": 1,
        "failures": [{
            "attempt": 1,
            "category": category,
            "failing_test": "tests/test_math.py::test_add",
            "evidence": (
                ["tests/test_math.py::test_add", "E AssertionError: expected 3, got 2"]
                if evidence is None else evidence
            ),
            "root_cause": "The implementation returned the wrong sum.",
            "root_cause_evidence": (
                ["E AssertionError: expected 3, got 2"]
                if root_cause_evidence is None else root_cause_evidence
            ),
            "repair_action": "Correct the addition implementation.",
        }],
    })


def test_schema_hint_requires_canonical_categories_and_json_lists():
    assert "incorrect_test" in gt.SCHEMA_HINT
    assert "application_bug" in gt.SCHEMA_HINT
    assert "evidence and root_cause_evidence must be JSON lists" in gt.SCHEMA_HINT
    assert "regression" in gt.SCHEMA_HINT


def test_normalize_maps_snake_case_categories_without_renaming_validator():
    report = json.loads(gt.normalize_diagnosis_report(
        _report(category="incorrect_test")
    ))

    assert report["failures"][0]["category"] == "incorrect test"


def test_normalize_string_evidence_into_list():
    report = json.loads(gt.normalize_diagnosis_report(
        _report(evidence="E AssertionError: expected 3, got 2")
    ))

    assert report["failures"][0]["evidence"] == [
        "E AssertionError: expected 3, got 2"
    ]


def test_valid_snake_case_report_with_list_evidence_is_accepted():
    result = gt.validate_repair_diagnosis_compatible(_report(), 1, OUTPUT)

    assert result["validated"] is True
    assert result["category"] == "application bug"
    assert result["repair_allowed"] is True


def test_string_evidence_is_accepted_after_normalization():
    report = _report(
        evidence="  FAILED tests/test_math.py::test_add  \n"
                 "  E AssertionError: expected 3, got 2  ",
        root_cause_evidence="E AssertionError: expected 3, got 2",
    )

    result = gt.validate_repair_diagnosis_compatible(report, 1, OUTPUT)

    assert result["validated"] is True
    assert result["repair_allowed"] is True


def test_whitespace_normalized_evidence_matches_captured_output():
    output = (
        "FAILED   tests/test_math.py::test_add\n"
        "E   AssertionError:   expected 3,  got 2"
    )
    report = _report(
        evidence="FAILED tests/test_math.py::test_add\nE AssertionError: expected 3, got 2",
        root_cause_evidence="E AssertionError: expected 3, got 2",
    )

    result = gt.validate_repair_diagnosis_compatible(report, 1, output)

    assert result["validated"] is True


def test_invalid_category_is_rejected():
    result = gt.validate_repair_diagnosis_compatible(
        _report(category="test_fixture"), 1, OUTPUT
    )

    assert result["validated"] is False
    assert "category" in result["decision_reason"]


def test_fabricated_evidence_is_rejected():
    report = _report(
        evidence=[
            "tests/test_math.py::test_add",
            "E AssertionError: expected 99, got 2",
        ],
        root_cause_evidence=["E AssertionError: expected 99, got 2"],
    )

    result = gt.validate_repair_diagnosis_compatible(report, 1, OUTPUT)

    assert result["validated"] is False


def test_schema_hint_is_appended_only_to_diagnosis_prompts():
    ordinary = "Please continue."
    diagnosis = "Diagnose only. Do not modify files."
    correction = "Replace the complete `failures` entry for attempt 2."

    assert gt.diagnosis_prompt_with_schema_hint(ordinary) == ordinary
    assert gt.SCHEMA_HINT in gt.diagnosis_prompt_with_schema_hint(diagnosis)
    assert gt.SCHEMA_HINT in gt.diagnosis_prompt_with_schema_hint(correction)


def _final_report(evidence, category="application_bug", **extra):
    report = {
        "schema_version": 1,
        "detected_stack": {},
        "environment": {"status": "ready"},
        "test_commands": ["pytest -q"],
        "failures": [{
            "attempt": 1,
            "category": category,
            "failing_test": "tests/test_math.py::test_add",
            "evidence": evidence,
            "root_cause": "The implementation returned the wrong sum.",
            "root_cause_evidence": ["E AssertionError: expected 3, got 2"],
            "repair_action": "Fix addition.",
        }],
        "dependency_installation_failures": [],
        "root_causes": [],
        "modified_files": [],
        "proposed_patches": [],
        "verified_test_results": [],
        "blockers": [],
    }
    report.update(extra)
    return report


def test_final_report_snake_case_category_is_normalized(tmp_path):
    report = _final_report(["tests/test_math.py::test_add"])
    normalized = json.loads(gt.normalize_final_agent_report(json.dumps(report), ATTEMPTS))

    assert normalized["failures"][0]["category"] == "application bug"


def test_final_report_string_evidence_becomes_list(tmp_path):
    report = _final_report("tests/test_math.py::test_add")
    normalized = json.loads(gt.normalize_final_agent_report(json.dumps(report), ATTEMPTS))

    assert normalized["failures"][0]["evidence"] == ["tests/test_math.py::test_add"]


def test_final_report_list_evidence_is_accepted(tmp_path):
    report = _final_report([
        "FAILED tests/test_math.py::test_add",
        "E AssertionError: expected 3, got 2",
    ])
    normalized = json.loads(gt.normalize_final_agent_report(json.dumps(report), ATTEMPTS))
    path = tmp_path / "openhands-report.json"
    path.write_text(json.dumps(normalized), encoding="utf-8")

    validated = load_agent_report(path, ATTEMPTS)

    assert validated["report_validation_errors"] == []
    assert validated["failures"][0]["category"] == "application bug"


def test_final_report_whitespace_normalized_evidence_is_accepted(tmp_path):
    report = _final_report(
        "  FAILED   tests/test_math.py::test_add  \n"
        "  E   AssertionError:   expected 3,   got 2  "
    )
    normalized = json.loads(gt.normalize_final_agent_report(json.dumps(report), ATTEMPTS))
    path = tmp_path / "openhands-report.json"
    path.write_text(json.dumps(normalized), encoding="utf-8")

    validated = load_agent_report(path, ATTEMPTS)

    assert validated["report_validation_errors"] == []
    assert "FAILED tests/test_math.py::test_add" in normalized["failures"][0]["evidence"]


def test_final_report_fabricated_evidence_is_still_rejected(tmp_path):
    report = _final_report([
        "tests/test_math.py::test_add",
        "E AssertionError: expected 99, got 2",
    ])
    normalized = json.loads(gt.normalize_final_agent_report(json.dumps(report), ATTEMPTS))
    path = tmp_path / "openhands-report.json"
    path.write_text(json.dumps(normalized), encoding="utf-8")

    validated = load_agent_report(path, ATTEMPTS)

    assert "E AssertionError: expected 99, got 2" not in normalized["failures"][0]["evidence"]
    assert any("does not match captured output" in error for error in validated["report_validation_errors"])


PASSING_ATTEMPTS = [{
    "attempt": 1,
    "exit_code": 0,
    "output": "135 passed in 4.90s\n",
}]


def test_generated_test_import_rule_documents_workspace_path_and_parents2():
    assert "/workspace/.github/scripts" in gt.GENERATED_TEST_IMPORT_RULE
    assert "parents[2]" in gt.GENERATED_TEST_IMPORT_RULE
    assert "import generate_tests" in gt.GENERATED_TEST_IMPORT_RULE


def test_report_evidence_rule_forbids_fixtures_and_fabricated_passing_failures():
    assert "verbatim" in gt.REPORT_EVIDENCE_RULE
    assert "fixtures" in gt.REPORT_EVIDENCE_RULE
    assert "failures: []" in gt.REPORT_EVIDENCE_RULE


def test_schema_hint_includes_both_prompt_rules():
    assert gt.GENERATED_TEST_IMPORT_RULE in gt.SCHEMA_HINT
    assert gt.REPORT_EVIDENCE_RULE in gt.SCHEMA_HINT


def test_fabricated_fixture_evidence_is_rejected_when_suite_passes(tmp_path):
    """Replay the CI failure: fixture text must not validate as evidence."""
    report = _final_report([
        "FAILED tests/generated/test_math.py::test_add",
        "E AssertionError: expected 3, got 2",
    ])

    normalized = json.loads(
        gt.normalize_final_agent_report(json.dumps(report), PASSING_ATTEMPTS)
    )
    assert normalized["failures"][0]["evidence"] == []

    path = tmp_path / "openhands-report.json"
    path.write_text(json.dumps(normalized), encoding="utf-8")
    validated = load_agent_report(path, PASSING_ATTEMPTS)

    assert any(
        "does not match captured output" in error
        for error in validated["report_validation_errors"]
    )
    assert any(
        "does not match a failed test run" in error
        for error in validated["report_validation_errors"]
    )
    assert validated["report_validation_errors"] != []


def test_final_report_normalization_preserves_unrelated_fields():
    report = _final_report(
        ["tests/test_math.py::test_add"],
        custom_marker="keep-me",
        blockers=["unrelated blocker"],
    )
    original = json.loads(json.dumps(report))
    normalized = json.loads(gt.normalize_final_agent_report(json.dumps(report), ATTEMPTS))

    assert normalized["custom_marker"] == "keep-me"
    assert normalized["blockers"] == ["unrelated blocker"]
    assert normalized["root_causes"] == original["root_causes"]
    assert normalized["verified_test_results"] == original["verified_test_results"]
    assert normalized["failures"][0]["root_cause_evidence"] == original["failures"][0]["root_cause_evidence"]


def test_schema_hint_requests_optional_verified_application_source_location():
    hint = gt.SCHEMA_HINT
    assert "affected_file" in hint
    assert "affected_line" in hint
    assert "application_bug" in hint
    assert "regression" in hint
    assert "null" in hint


def test_validated_application_diagnosis_copies_source_location():
    report = {
        "failures": [{
            "attempt": 1,
            "category": "unknown",
            "affected_file": "untrusted/raw.py",
            "affected_line": 999,
        }]
    }
    attempts = [{
        "attempt": 1,
        "diagnosis": {
            "validated": True,
            "category": "application bug",
            "evidence": ["E AssertionError: expected 3, got 2"],
            "root_cause": "Incorrect addition result.",
            "failing_test": "tests/test_math.py::test_add",
            "root_cause_evidence": ["E AssertionError: expected 3, got 2"],
            "affected_file": "src/math.py",
            "affected_line": 42,
        },
    }]

    gates = gt.apply_validated_diagnoses(report, attempts)

    failure = report["failures"][0]
    assert gates[1]["diagnosis"]["validated"] is True
    assert failure["category"] == "application bug"
    assert failure["affected_file"] == "src/math.py"
    assert failure["affected_line"] == 42


def test_unvalidated_diagnosis_cannot_publish_raw_source_location():
    report = {
        "failures": [{
            "attempt": 1,
            "category": "application_bug",
            "affected_file": "untrusted/raw.py",
            "affected_line": 999,
        }]
    }
    attempts = [{
        "attempt": 1,
        "diagnosis": {
            "validated": False,
            "category": "application bug",
            "affected_file": "untrusted/raw.py",
            "affected_line": 999,
        },
    }]

    gt.apply_validated_diagnoses(report, attempts)

    failure = report["failures"][0]
    assert failure["category"] == "unknown"
    assert "affected_file" not in failure
    assert "affected_line" not in failure


def test_validated_non_application_diagnosis_does_not_copy_source_location():
    report = {
        "failures": [{
            "attempt": 1,
            "category": "unknown",
            "affected_file": "untrusted/raw.py",
            "affected_line": 999,
        }]
    }
    attempts = [{
        "attempt": 1,
        "diagnosis": {
            "validated": True,
            "category": "incorrect test",
            "affected_file": "tests/test_math.py",
            "affected_line": 12,
        },
    }]

    gt.apply_validated_diagnoses(report, attempts)

    failure = report["failures"][0]
    assert failure["category"] == "incorrect test"
    assert "affected_file" not in failure
    assert "affected_line" not in failure


def test_validated_application_diagnosis_without_location_remains_valid():
    report = {
        "failures": [{
            "attempt": 1,
            "category": "unknown",
            "affected_file": "untrusted/raw.py",
            "affected_line": 999,
        }]
    }
    attempts = [{
        "attempt": 1,
        "diagnosis": {
            "validated": True,
            "category": "regression",
            "evidence": ["E AssertionError: expected 3, got 2"],
            "root_cause": "A regression changed the result.",
            "failing_test": "tests/test_math.py::test_add",
            "root_cause_evidence": ["E AssertionError: expected 3, got 2"],
            "affected_file": None,
            "affected_line": None,
        },
    }]

    gt.apply_validated_diagnoses(report, attempts)

    failure = report["failures"][0]
    assert failure["category"] == "regression"
    assert "affected_file" not in failure
    assert "affected_line" not in failure


def test_agent_validation_flags_cannot_override_invalid_evidence(tmp_path):
    report = _final_report(["E AssertionError: expected 99, got 2"])
    report.update({
        "agent_report_valid": True,
        "report_valid": True,
        "report_validation_errors": [],
    })
    normalized = json.loads(gt.normalize_final_agent_report(json.dumps(report), ATTEMPTS))
    assert normalized["agent_report_valid"] is True
    assert normalized["report_validation_errors"] == []

    path = tmp_path / "openhands-report.json"
    path.write_text(json.dumps(normalized), encoding="utf-8")
    validated = load_agent_report(path, ATTEMPTS)

    assert "agent_report_valid" not in validated
    assert "report_valid" not in validated
    assert any(
        "does not match captured output" in error
        for error in validated["report_validation_errors"]
    )
    gt.finalize_report_validity(validated, True)
    assert validated["agent_report_valid"] is False
    assert validated["report_valid"] is False


def test_agent_empty_validation_errors_cannot_hide_schema_failure(tmp_path):
    report = _final_report(["tests/test_math.py::test_add"])
    del report["blockers"]
    report.update({"report_valid": True, "report_validation_errors": []})

    normalized = json.loads(gt.normalize_final_agent_report(json.dumps(report), ATTEMPTS))
    path = tmp_path / "openhands-report.json"
    path.write_text(json.dumps(normalized), encoding="utf-8")
    validated = load_agent_report(path, ATTEMPTS)

    assert any(
        "missing required field: blockers" in error
        for error in validated["report_validation_errors"]
    )
    gt.finalize_report_validity(validated, True)
    assert validated["agent_report_valid"] is False
    assert validated["report_valid"] is False


def test_missing_or_malformed_report_fails_closed(tmp_path):
    path = tmp_path / "openhands-report.json"
    for content in (None, "not json", "[]", '{"schema_version": 1}'):
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(content, encoding="utf-8")

        validated = load_agent_report(path, ATTEMPTS)
        assert validated["report_validation_errors"]
        assert validated["failures"] == []
        gt.finalize_report_validity(validated, True)
        assert validated["agent_report_valid"] is False
        assert validated["report_valid"] is False

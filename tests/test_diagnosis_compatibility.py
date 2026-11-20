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

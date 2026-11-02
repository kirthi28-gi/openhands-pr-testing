import pytest
from Calculator import SafeEvaluator

def test_ceil_positive_decimal():
    ev = SafeEvaluator()
    assert ev.evaluate("ceil(2.3)") == 3, f"Expected 3, got {ev.evaluate('ceil(2.3)')}"

def test_ceil_whole_number():
    ev = SafeEvaluator()
    result = ev.evaluate("ceil(5.0)")
    assert result == 5, f"Expected 5, got {result}"

def test_ceil_negative_decimal():
    ev = SafeEvaluator()
    result = ev.evaluate("ceil(-2.3)")
    assert result == -2, f"Expected -2, got {result}"

def test_ceil_zero():
    ev = SafeEvaluator()
    assert ev.evaluate("ceil(0)") == 0

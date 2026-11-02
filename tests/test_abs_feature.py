import math
import pytest
from Calculator import SafeEvaluator

def test_abs_positive_integer():
    ev = SafeEvaluator()
    assert ev.evaluate("abs(5)") == 5

def test_abs_negative_integer():
    ev = SafeEvaluator()
    result = ev.evaluate("abs(-5)")
    assert result == 5, f"Expected 5, got {result}"

def test_abs_zero():
    ev = SafeEvaluator()
    assert ev.evaluate("abs(0)") == 0

def test_abs_negative_decimal():
    ev = SafeEvaluator()
    result = ev.evaluate("abs(-3.5)")
    assert result == 3.5, f"Expected 3.5, got {result}"

import math
import pytest
from Calculator import SafeEvaluator


class TestAdditionalPositiveCases:
    def test_division_positive(self):
        ev = SafeEvaluator()
        assert ev.evaluate("10/2") == 5.0
        assert ev.evaluate("10/3") == 3.3333333333333335

    def test_precedence_mixed(self):
        ev = SafeEvaluator()
        assert ev.evaluate("2+3*4-6/2") == 11.0

    def test_unary_negative_and_double_neg(self):
        ev = SafeEvaluator()
        assert ev.evaluate("-5") == -5
        assert ev.evaluate("--5") == 5
        assert ev.evaluate("+5") == 5

    def test_nested_functions(self):
        ev = SafeEvaluator()
        assert ev.evaluate("sqrt(sqrt(16))") == 2.0

    def test_trig_full_circle(self):
        ev = SafeEvaluator()
        assert math.isclose(ev.evaluate("sin(30)"), 0.5, rel_tol=1e-9)
        assert math.isclose(ev.evaluate("sin(90)"), 1.0, rel_tol=1e-9)
        assert math.isclose(ev.evaluate("cos(0)"), 1.0, rel_tol=1e-9)
        assert math.isclose(ev.evaluate("tan(45)"), 1.0, rel_tol=1e-9)
        assert math.isclose(ev.evaluate("sin(0)"), 0.0, rel_tol=1e-9)

    def test_constants_in_expression(self):
        ev = SafeEvaluator()
        assert math.isclose(ev.evaluate("2*pi"), 2 * math.pi, rel_tol=1e-12)
        assert math.isclose(ev.evaluate("pi + e"), math.pi + math.e, rel_tol=1e-12)

    def test_sqrt_and_log_edge_positive(self):
        ev = SafeEvaluator()
        assert ev.evaluate("sqrt(0)") == 0.0
        assert math.isclose(ev.evaluate("log(1)"), 0.0, rel_tol=1e-9)


class TestAdditionalNegativeCases:
    def test_division_by_zero_raises_zero_division(self):
        ev = SafeEvaluator()
        with pytest.raises(ZeroDivisionError):
            ev.evaluate("1/0")

    def test_sqrt_negative_raises_valueerror(self):
        ev = SafeEvaluator()
        with pytest.raises(ValueError):
            ev.evaluate("sqrt(-1)")

    def test_log_non_positive_raises_valueerror(self):
        ev = SafeEvaluator()
        with pytest.raises(ValueError):
            ev.evaluate("log(0)")
        with pytest.raises(ValueError):
            ev.evaluate("log(-5)")

    def test_invalid_identifier_raises(self):
        ev = SafeEvaluator()
        with pytest.raises(ValueError):
            ev.evaluate("unknown_name")
        with pytest.raises(ValueError):
            ev.evaluate("foo")

    def test_invalid_function_raises(self):
        ev = SafeEvaluator()
        with pytest.raises(ValueError):
            ev.evaluate("factorial(5)")

    def test_empty_function_call_raises(self):
        ev = SafeEvaluator()
        with pytest.raises(TypeError):
            ev.evaluate("sin()")

    def test_too_many_args_raises(self):
        ev = SafeEvaluator()
        with pytest.raises(TypeError):
            ev.evaluate("sin(30, 45)")

    def test_malformed_syntax_raises(self):
        ev = SafeEvaluator()
        with pytest.raises(SyntaxError):
            ev.evaluate("2+")


class TestAdditionalEdgeCases:
    def test_tan_near_asymptote_large(self):
        ev = SafeEvaluator()
        result = ev.evaluate("tan(90)")
        assert isinstance(result, float)
        assert abs(result) > 1e15

    def test_power_zero_and_large(self):
        ev = SafeEvaluator()
        assert ev.evaluate("5^0") == 1
        assert ev.evaluate("0^5") == 0

    def test_modulo_negative_divisor(self):
        ev = SafeEvaluator()
        assert ev.evaluate("10%-3") == -2

    def test_chained_trig(self):
        ev = SafeEvaluator()
        assert math.isclose(
            ev.evaluate("sin(cos(30))"), math.sin(math.radians(math.cos(math.radians(30)))), rel_tol=1e-9
        )

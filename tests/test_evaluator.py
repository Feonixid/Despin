import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load("despin_evaluator", "src/despin/evaluator.py")
E = ev.RobustMathEvaluator


def test_dfrac_equals_frac_in_tuple():
    # math500_0: scored WRONG by the legacy evaluator for every method
    assert E.is_equiv(r"\left(3,\dfrac{\pi}{2}\right)", r"\left( 3, \frac{\pi}{2} \right)")


def test_percent_matches_bare_number():
    # gsm8k_14: gold 60, answer "60%" was scored wrong by the legacy evaluator
    assert E.is_equiv(r"60\%", "60")


def test_tuple_commas_are_not_collapsed():
    assert not E.is_equiv("1,-2", "-1")  # legacy stripped the comma and compared 1-2 == -1
    assert E.is_equiv("1,2", "1,2")
    assert not E.is_equiv("1,2", "12")


def test_thousands_separator():
    assert E.is_equiv("1,000", "1000")


def test_truncated_boxed_is_no_answer():
    assert ev.extract_boxed(r"so we get \boxed{3\sqrt{13") is None
    assert E.extract_answer(r"unfinished reasoning ... the answer is 5") is None


def test_nested_boxed_and_fraction_forms():
    assert ev.extract_boxed(r"\boxed{\frac{14}{3}}") == r"\frac{14}{3}"
    assert E.is_equiv(r"\frac{14}{3}", "14/3")
    assert E.is_equiv(r"-\frac{1}{2}", "-0.5")


def test_units_after_number():
    assert E.is_equiv(r"18\text{ dollars}", "18")
    assert not E.is_equiv(r"18\text{ dollars}", "19")


def test_fallback_is_opt_in():
    assert E.extract_answer("so 42", allow_fallback=True) == "42"
    assert E.extract_answer("so 42") is None


def test_stats_helpers():
    ma = _load("make_paper_assets", "scripts/make_paper_assets.py")
    lo, hi = ma.wilson(36, 60)
    assert abs(lo - 0.474) < 0.002 and abs(hi - 0.714) < 0.002
    a, b, p = ma.mcnemar_exact([1] * 4 + [0] * 56, [0] * 60)
    assert (a, b) == (4, 0) and abs(p - 0.125) < 1e-9

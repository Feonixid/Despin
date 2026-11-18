"""Stricter, normalizing answer checker for NEW runs.

The scorer that produced the numbers in the paper is preserved unchanged in
``legacy/logged_run/despin/evaluator.py``. Known problems of that scorer (see paper, Section 5.6):
it strips every comma and closing brace, does not unify \\dfrac/\\frac, does not match "60%" to "60",
and falls back to heuristics (last number on the last line) when no \\boxed{} exists, which lets
unfinished generations be scored. This module fixes those behaviours:

* the answer must be a complete ``\\boxed{...}`` with balanced braces (no heuristic fallback unless
  ``allow_fallback=True`` is requested explicitly);
* LaTeX normalization (dfrac/tfrac, \\left/\\right, \\text{}, degrees, percent, spacing) keeps commas
  inside tuples/lists, and removes only thousands separators;
* numeric comparison understands ``a/b`` and ``\\frac{a}{b}``.
"""
import re
from typing import Optional


def extract_boxed(text: str) -> Optional[str]:
    """Contents of the last ``\\boxed{...}`` with balanced braces, or None (also None if truncated)."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx + len("\\boxed{")
    depth, out = 1, []
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(c)
        i += 1
    return None


def normalize(s: str) -> str:
    s = s.strip().replace("$", "")
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = re.sub(r"\\(left|right)", "", s)
    s = re.sub(r"\\[,!; ]", "", s)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "%").replace("\\$", "")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)  # thousands separators only
    s = re.sub(r"\s+", "", s)
    return s.rstrip(".")


_NUM = r"-?\d+(?:\.\d+)?"


def to_number(s: str) -> float:
    s = s.rstrip("%")
    m = re.fullmatch(rf"(-?)\\frac\{{({_NUM})\}}\{{({_NUM})\}}", s)
    if m:
        v = float(m.group(2)) / float(m.group(3))
        return -v if m.group(1) else v
    m = re.fullmatch(rf"({_NUM})/({_NUM})", s)
    if m:
        return float(m.group(1)) / float(m.group(2))
    return float(s)


class RobustMathEvaluator:
    @staticmethod
    def extract_answer(text: str, allow_fallback: bool = False) -> Optional[str]:
        ans = extract_boxed(text or "")
        if ans is not None or not allow_fallback:
            return ans
        lines = [ln.strip() for ln in (text or "").strip().split("\n") if ln.strip()]
        if lines:
            m = re.search(r"[-+]?\d*\.?\d+(?:/\d+)?", lines[-1])
            if m:
                return m.group(0)
        return None

    @staticmethod
    def is_equiv(pred: Optional[str], ground_truth: Optional[str]) -> bool:
        if pred is None or ground_truth is None:
            return False
        p, g = normalize(pred), normalize(ground_truth)
        if not p or not g:
            return False
        if p.lower() == g.lower():
            return True
        # "18dollars" -> "18": a leading number followed only by letters is read as number + unit
        m = re.fullmatch(rf"({_NUM})[A-Za-z]+", p)
        if m:
            p = m.group(1)
        try:
            return abs(to_number(p) - to_number(g)) < 1e-6
        except (ValueError, ZeroDivisionError):
            return False

import re
from typing import Optional, Any
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


class MathEvaluator:
    """
    Extracts and rigorously verifies mathematical answers for MATH and GSM8K benchmarks.
    Handles LaTeX boxed expressions, fractions, decimals, units, and symbolic equivalence via SymPy.
    """

    BOXED_REGEX = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
    ANSWER_PATTERNS = [
        re.compile(r"(?i)the (?:final )?answer is:?\s*([^\n\.]+)"),
        re.compile(r"(?i)####\s*([^\n\.]+)"),
        re.compile(r"(?i)therefore,?\s*(?:the answer is)?\s*([^\n\.]+)"),
        re.compile(r"(?i)=\s*([0-9\/\.\-]+)\s*$"),
    ]

    @classmethod
    def extract_answer(cls, text: str) -> Optional[str]:
        if not text:
            return None

        # 1. LaTeX \boxed{...}
        boxed_matches = cls.BOXED_REGEX.findall(text)
        if boxed_matches:
            return cls._clean_answer_string(boxed_matches[-1])

        # 2. Explicit patterns
        for pattern in cls.ANSWER_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                candidate = matches[-1].strip().rstrip(".")
                return cls._clean_answer_string(candidate)

        # 3. Fallback: Last numeric line
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if lines:
            last_line = lines[-1]
            num_match = re.search(r"[-+]?\d*\.?\d+(?:/\d+)?", last_line)
            if num_match:
                return num_match.group(0).strip()

        return None

    @classmethod
    def _clean_answer_string(cls, ans: str) -> str:
        if not ans:
            return ""
        s = ans.strip()
        s = s.replace("$", "").replace(" ", "")
        s = s.replace(r"\text{", "").replace("}", "")
        s = s.replace(r"\%", "%").replace(r"\$", "$")
        s = s.replace(r"^\circ", "").replace(r"^{\circ}", "").replace("degrees", "").replace("degree", "")
        # Remove LaTeX backslashes before digits
        s = re.sub(r"\\(\d)", r"\1", s)
        s = s.replace(r"\!", "").replace(",", "")

        # 1. If string begins with a distinct number followed by words (e.g. "27isthesmallest..."), extract the number
        lead_num = re.match(r"^([-+]?\d+(?:\.\d+)?)(?:[a-zA-Z]|$)", s)
        if lead_num:
            return lead_num.group(1)

        # 2. Extract trailing expression if preceded by words (e.g. "thedoublesumisp-q" -> "p-q")
        m_trailing = re.search(r"(?:is|=)([a-zA-Z0-9\+\-\*/\^]+)$", s)
        if m_trailing:
            s = m_trailing.group(1)

        # 3. Fix malformed LaTeX fractions e.g. \dfrac{14{3 -> 14/3
        s = re.sub(r"\\d?frac\{(\d+)\{(\d+)", r"\1/\2", s)
        s = re.sub(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
        return s.rstrip(".")

    @classmethod
    def is_equiv(cls, pred: Optional[str], ground_truth: str) -> bool:
        if pred is None or ground_truth is None:
            return False

        p_str = cls._clean_answer_string(pred)
        gt_str = cls._clean_answer_string(ground_truth)

        if not p_str or not gt_str:
            return False

        # 1. Exact string match
        if p_str.lower() == gt_str.lower():
            return True

        # 2. Numeric float equivalence
        try:
            def to_float(val: str) -> float:
                if "/" in val and not re.search(r"[a-zA-Z]", val):
                    num, den = val.split("/")
                    return float(num) / float(den)
                return float(val)

            p_val = to_float(p_str)
            g_val = to_float(gt_str)
            if abs(p_val - g_val) < 1e-4:
                return True
        except Exception:
            pass

        # 3. Symbolic Equivalence via SymPy
        try:
            transformations = standard_transformations + (implicit_multiplication_application,)
            expr_p = parse_expr(p_str, transformations=transformations)
            expr_g = parse_expr(gt_str, transformations=transformations)
            diff = sympy.simplify(expr_p - expr_g)
            if diff == 0:
                return True
        except Exception:
            pass

        return False

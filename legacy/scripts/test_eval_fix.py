import re
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class ImprovedMathEvaluator:
    BOXED_REGEX = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

    @classmethod
    def clean_text(cls, s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        s = s.replace("$", "").replace(" ", "")
        s = s.replace(r"\text{", "").replace("}", "")
        s = s.replace(r"\%", "%").replace(r"\$", "$")
        s = s.replace(r"^\circ", "").replace(r"^{\circ}", "").replace("degrees", "").replace("degree", "")
        # Remove LaTeX backslashes before digits (e.g. \460 -> 460, \195,\!000 -> 195000)
        s = re.sub(r"\\(\d)", r"\1", s)
        s = s.replace(r"\!", "").replace(",", "")
        # Extract trailing equation / expression if preceded by words (e.g. "thedoublesumisp-q" -> "p-q")
        m = re.search(r"is([a-zA-Z0-9\+\-\*/\^]+)$", s)
        if m:
            s = m.group(1)
        # Extract leading number if followed by words (e.g. "27isthesmallest..." -> "27")
        m_num = re.match(r"^([-+]?\d*\.?\d+(?:/\d+)?)[a-zA-Z]", s)
        if m_num:
            s = m_num.group(1)
        # Handle malformed \dfrac{a{b -> a/b
        s = re.sub(r"\\d?frac\{(\d+)\{(\d+)", r"\1/\2", s)
        s = re.sub(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
        return s.rstrip(".")

    @classmethod
    def is_equiv(cls, pred: str, gt: str) -> bool:
        if pred is None or gt is None:
            return False
        p_clean = cls.clean_text(pred)
        g_clean = cls.clean_text(gt)

        if not p_clean or not g_clean:
            return False

        if p_clean.lower() == g_clean.lower():
            return True

        # Number check
        try:
            def to_f(v):
                if "/" in v and not re.search(r"[a-zA-Z]", v):
                    nums = v.split("/")
                    return float(nums[0]) / float(nums[1])
                return float(v)
            if abs(to_f(p_clean) - to_f(g_clean)) < 1e-4:
                return True
        except Exception:
            pass

        # Sympy check
        try:
            trans = standard_transformations + (implicit_multiplication_application,)
            ep = parse_expr(p_clean, transformations=trans)
            eg = parse_expr(g_clean, transformations=trans)
            if sympy.simplify(ep - eg) == 0:
                return True
        except Exception:
            pass

        return False

# Test on the problem cases
cases = [
    (r"\460", "460"),
    ("90", r"90^\circ"),
    ("thedoublesumisp-q", "p - q"),
    ("27isthesmallestpositiveperfectcubethatcanbewrittenasthesumofthreeconsecutiveintegers", "27"),
    (r"\dfrac{14{3", r"\frac{14}{3}"),
    (r"\left(3,\dfrac{\pi{2\right)", r"\left( 3, \frac{\pi}{2} \right)"),
]

for p, g in cases:
    print(f"Pred: {p:30s} | GT: {g:15s} -> Equiv: {ImprovedMathEvaluator.is_equiv(p, g)}")


#!/usr/bin/env python3
"""Re-derive every number, table and figure-data file used in the paper from the logged JSON.

Usage (from the repository root):
    python scripts/make_paper_assets.py \
        --input data/logged_run/massive_sweep_all_baselines.json \
        --out-dir paper/generated --summary results/analysis_summary.json

Nothing in the paper is typed in by hand: tables are written to paper/generated/*.tex and every
in-text statistic is exported as a LaTeX macro in paper/generated/stats.tex.
"""
import argparse
import json
import math
import os
from collections import Counter

from scipy.stats import binomtest

CAP = 2048  # max_new_tokens used in every run
METHODS = [  # (json key, display name, short macro name)
    ("greedy", "Greedy ($T{=}0$)", "Greedy"),
    ("sampling", "Temperature sampling ($T{=}0.6$)", "Samp"),
    ("repetition_penalty", "Repetition penalty ($\\theta{=}1.15$)", "Rep"),
    ("passive_early_exit", "Detector-triggered early exit", "Early"),
    ("despin", "DESPIN (as implemented)", "Despin"),
]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def mcnemar_exact(a, b):
    """a, b: lists of 0/1. Returns (only_a, only_b, two-sided exact p)."""
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return only_a, only_b, p


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["records"]


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def despin_events(rec):
    hist = rec["despin"].get("rollback_history", [])
    asd = [h for h in hist if h["type"] == "asd_rollback"]
    pgec = [h for h in hist if h["type"] == "pgec_early_closure"]
    first = hist[0] if hist else None
    dup = sum(
        1
        for a, b in zip(asd, asd[1:])
        if (a["at_token"], a["rewound_to"], a["reason"]) == (b["at_token"], b["rewound_to"], b["reason"])
    )
    ftype = None
    if first is not None:
        ftype = "R" if "Recurrence" in first["reason"] else "V"
    return asd, pgec, first, dup, ftype


def analyse(records):
    n = len(records)
    S = {"n": n}
    keys = [k for k, _, _ in METHODS]
    corr = {k: [int(r[k]["correct"]) for r in records] for k in keys}
    tok = {k: [r[k]["tokens"] for r in records] for k in keys}
    is_math = [r["problem"]["benchmark"] == "MATH-500" for r in records]
    pruned = [r["despin"].get("pruned", 0) for r in records]

    S["methods"] = {}
    for k, name, short in METHODS:
        c = sum(corr[k])
        lo, hi = wilson(c, n)
        total_tok = sum(tok[k]) + (sum(pruned) if k == "despin" else 0)
        S["methods"][k] = dict(
            name=name, short=short, correct=c, acc=100 * c / n, ci_lo=100 * lo, ci_hi=100 * hi,
            math_correct=sum(x for x, m in zip(corr[k], is_math) if m),
            gsm_correct=sum(x for x, m in zip(corr[k], is_math) if not m),
            avg_final_tokens=sum(tok[k]) / n, avg_total_tokens=total_tok / n,
            tokens_per_correct=total_tok / max(c, 1),
            cap_hits=sum(1 for t in tok[k] if t >= CAP),
            correct_and_capped=sum(1 for x, t in zip(corr[k], tok[k]) if x and t >= CAP),
            tes_final=(100 * c / n) / (sum(tok[k]) / n / 1000),
            tes_total=(100 * c / n) / (total_tok / n / 1000),
        )
    S["mcnemar"] = {}
    for k in keys:
        if k != "greedy":
            a, b, p = mcnemar_exact(corr[k], corr["greedy"])
            S["mcnemar"][k + "_vs_greedy"] = dict(only_method=a, only_greedy=b, p=p)
    for k in ("sampling", "repetition_penalty"):
        a, b, p = mcnemar_exact(corr["despin"], corr[k])
        S["mcnemar"]["despin_vs_" + k] = dict(only_despin=a, only_other=b, p=p)

    S["rescues"] = [
        dict(id=r["problem"]["id"], subject=r["problem"]["subject"], level=r["problem"]["level"],
             greedy_tokens=r["greedy"]["tokens"], despin_tokens=r["despin"]["tokens"],
             pruned=r["despin"]["pruned"], capped=r["despin"]["tokens"] >= CAP)
        for r in records if r["despin"]["correct"] and not r["greedy"]["correct"]
    ]
    S["regressions"] = sum(1 for r in records if r["greedy"]["correct"] and not r["despin"]["correct"])

    ev = [despin_events(r) for r in records]
    trig = [bool(e[0] or e[1]) for e in ev]
    S["trig_any"] = sum(trig)
    S["trig_math"] = sum(t for t, m in zip(trig, is_math) if m)
    S["trig_gsm"] = sum(t for t, m in zip(trig, is_math) if not m)
    healthy = [bool(r["greedy"]["correct"] and r["greedy"]["tokens"] < CAP) for r in records]
    capped = [r["greedy"]["tokens"] >= CAP for r in records]
    S["healthy_n"] = sum(healthy)
    S["healthy_trig"] = sum(h and t for h, t in zip(healthy, trig))
    S["greedy_capped_n"] = sum(capped)
    S["greedy_capped_trig"] = sum(c and t for c, t in zip(capped, trig))
    fired = [r["passive_early_exit"]["tokens"] < r["greedy"]["tokens"] for r in records]
    ee = [r["passive_early_exit"]["tokens"] for r, f in zip(records, fired) if f]
    S["early_truncated"] = sum(fired)
    S["early_median_len"] = median(ee)
    S["early_max_len"] = max(ee)
    first_pos = [e[2]["at_token"] for e in ev if e[2] is not None]
    S["first_trig_n"] = len(first_pos)
    S["first_trig_median"] = median(first_pos)
    S["first_trig_mean"] = sum(first_pos) / len(first_pos)
    S["first_trig_early"] = sum(1 for x in first_pos if x <= 220)
    S["first_types"] = dict(Counter(e[4] for e in ev if e[4]))
    S["asd_problems"] = sum(1 for e in ev if e[0])
    S["asd_at_cap"] = sum(1 for e in ev if len(e[0]) >= 2)
    S["dup_problems"] = sum(1 for e in ev if e[3])
    S["dup_wasted"] = sum(p for p, e in zip(pruned, ev) if e[3])
    S["pruned_total"] = sum(pruned)
    S["pruned_mean"] = sum(pruned) / n
    S["pruned_in_capped_despin"] = sum(p for p, r in zip(pruned, records) if r["despin"]["tokens"] >= CAP)
    S["despin_capped_n"] = sum(1 for r in records if r["despin"]["tokens"] >= CAP)
    S["noop_ids"] = [r["problem"]["id"] for r, e in zip(records, ev)
                     if e[0] and r["despin"]["tokens"] == r["greedy"]["tokens"] < CAP]
    pg = [r["despin"]["tokens"] - r["greedy"]["tokens"] for r, e in zip(records, ev) if e[1]]
    S["pgec_n"] = len(pg)
    S["pgec_deltas"] = pg
    S["pgec_mean_delta"] = sum(pg) / max(len(pg), 1)
    S["pgec_shorter"] = sum(1 for d in pg if d < 0)
    S["len_changed"] = sum(1 for r in records if r["despin"]["tokens"] != r["greedy"]["tokens"])
    S["len_changed_trig"] = sum(1 for r, t in zip(records, trig) if t and r["despin"]["tokens"] != r["greedy"]["tokens"])
    mathrec = [r for r in records if r["problem"]["benchmark"] == "MATH-500"]
    S["subjects"] = dict(Counter(r["problem"]["subject"] for r in mathrec))
    S["levels"] = {str(k): v for k, v in sorted(Counter(r["problem"]["level"] for r in mathrec).items())}
    S["_first_pos"] = first_pos
    return S


def esc(s):
    return str(s).replace("_", "\\_").replace("&", "\\&")


def camel(key):
    return "".join(w.capitalize() for w in key.split("_"))


def write_stats_macros(S, path):
    m = S["methods"]
    lines = ["% AUTO-GENERATED by scripts/make_paper_assets.py -- do not edit by hand."]

    def mac(name, val):
        lines.append("\\newcommand{\\%s}{%s}" % (name, val))

    for k, d in m.items():
        s = d["short"]
        mac(f"stat{s}Correct", d["correct"])
        mac(f"stat{s}Acc", f"{d['acc']:.1f}")
        mac(f"stat{s}CILo", f"{d['ci_lo']:.1f}")
        mac(f"stat{s}CIHi", f"{d['ci_hi']:.1f}")
        mac(f"stat{s}MathCorrect", d["math_correct"])
        mac(f"stat{s}GsmCorrect", d["gsm_correct"])
        mac(f"stat{s}AvgFinal", f"{d['avg_final_tokens']:,.1f}")
        mac(f"stat{s}AvgTotal", f"{d['avg_total_tokens']:,.1f}")
        mac(f"stat{s}TokPerCorrect", f"{d['tokens_per_correct']:,.0f}")
        mac(f"stat{s}CapHits", d["cap_hits"])
        mac(f"stat{s}CorrectCapped", d["correct_and_capped"])
        mac(f"stat{s}TESFinal", f"{d['tes_final']:.2f}")
        mac(f"stat{s}TESTotal", f"{d['tes_total']:.2f}")
    for key, v in S["mcnemar"].items():
        k2 = camel(key)
        mac(f"statP{k2}", f"{v['p']:.3f}")
        for sub, val in v.items():
            if sub != "p":
                mac(f"statN{k2}{camel(sub)}", val)
    simple = ["trig_any", "trig_math", "trig_gsm", "healthy_n", "healthy_trig", "greedy_capped_n",
              "greedy_capped_trig", "early_truncated", "early_median_len", "early_max_len", "first_trig_n",
              "first_trig_median", "first_trig_early", "asd_problems", "asd_at_cap", "dup_problems", "dup_wasted",
              "pruned_total", "pruned_in_capped_despin", "despin_capped_n", "pgec_n", "pgec_shorter",
              "len_changed", "len_changed_trig", "regressions"]
    for k in simple:
        v = S[k]
        mac("stat" + camel(k), f"{v:,}" if isinstance(v, int) else (f"{v:g}"))
    mac("statFirstTrigMean", f"{S['first_trig_mean']:.0f}")
    mac("statPrunedMean", f"{S['pruned_mean']:.1f}")
    mac("statPgecMeanDelta", f"{abs(S['pgec_mean_delta']):.1f}")
    mac("statNoopCount", len(S["noop_ids"]))
    mac("statVelFirst", S["first_types"].get("V", 0))
    mac("statRecFirst", S["first_types"].get("R", 0))
    mac("statRescueCapped", sum(1 for r in S["rescues"] if r["capped"]))
    mac("statRescueCount", len(S["rescues"]))
    mac("statTotalRatio", f"{100 * (m['despin']['avg_total_tokens'] / m['greedy']['avg_final_tokens'] - 1):+.1f}")
    mac("statFinalRatio", f"{100 * (m['despin']['avg_final_tokens'] / m['greedy']['avg_final_tokens'] - 1):+.1f}")
    mac("statTokPerCorrectGain",
        f"{100 * (1 - m['despin']['tokens_per_correct'] / m['greedy']['tokens_per_correct']):.1f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_main_table(S, path):
    rows = []
    for k, name, _ in METHODS:
        d = S["methods"][k]
        pv = None if k == "greedy" else S['mcnemar'][k + '_vs_greedy']['p']
        p = "--" if pv is None else ("$<$0.001" if pv < 0.001 else f"{pv:.3f}")
        rows.append(
            f"{name} & {d['math_correct']}/30 & {d['gsm_correct']}/30 & {d['correct']}/60 "
            f"({d['acc']:.1f}\\%) & [{d['ci_lo']:.1f}, {d['ci_hi']:.1f}] & {d['avg_final_tokens']:,.0f} & "
            f"{d['avg_total_tokens']:,.0f} & {d['tokens_per_correct']:,.0f} & {p} \\\\")
    head = ("\\begin{tabular}{lcccccccc}\n\\toprule\n"
            "Method & MATH & GSM8K & Overall & 95\\% CI & Final & Total & Tok/corr. & $p$ vs greedy \\\\\n\\midrule\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("% AUTO-GENERATED\n" + head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def write_perproblem(records, path):
    def mark(c):
        return "\\ding{51}" if c else "\\ding{55}"

    rows = []
    for r in records:
        p = r["problem"]
        asd, pg, _, dup, _ = despin_events(r)
        ev = f"{len(asd)}{'+P' if pg else ''}{'*' if dup else ''}"
        sub = esc(p["subject"].replace("Intermediate Algebra", "Int. Algebra")
                  .replace("Counting & Probability", "Count. & Prob."))
        lvl = p["level"] if p["benchmark"] == "MATH-500" else "--"
        rows.append(
            f"{esc(p['id'])} & {sub} & {lvl} & {mark(r['greedy']['correct'])} {r['greedy']['tokens']} & "
            f"{mark(r['sampling']['correct'])} {r['sampling']['tokens']} & "
            f"{mark(r['repetition_penalty']['correct'])} {r['repetition_penalty']['tokens']} & "
            f"{mark(r['passive_early_exit']['correct'])} {r['passive_early_exit']['tokens']} & "
            f"{mark(r['despin']['correct'])} {r['despin']['tokens']} & {r['despin']['pruned']} & {ev} \\\\")
    head = ("{\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{llcccccccc}\n\\toprule\n"
            "Problem & Subject & L & Greedy & Sampl. & Rep.\\ pen. & Early & DESPIN & Pruned & Ev \\\\\n\\midrule\n\\endhead\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("% AUTO-GENERATED\n" + head + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n}\n")


def write_figures(S, records, outdir):
    coords = []
    for i, (k, _, _) in enumerate(METHODS):
        d = S["methods"][k]
        coords.append(f"({i},{d['acc']:.2f}) += (0,{d['ci_hi'] - d['acc']:.2f}) -= (0,{d['acc'] - d['ci_lo']:.2f})")
    with open(os.path.join(outdir, "fig_acc_data.tex"), "w") as f:
        f.write("% AUTO-GENERATED\n\\addplot+[ybar, error bars/.cd, y dir=both, y explicit] coordinates {\n"
                + "\n".join(coords) + "\n};\n")
    rescued, same, diff = [], [], []
    for r in records:
        g, d = r["greedy"]["tokens"], r["despin"]["tokens"]
        pt = f"({g},{d})"
        if r["despin"]["correct"] and not r["greedy"]["correct"]:
            rescued.append(pt)
        elif g == d:
            same.append(pt)
        else:
            diff.append(pt)
    with open(os.path.join(outdir, "fig_scatter_data.tex"), "w") as f:
        f.write("% AUTO-GENERATED\n")
        for label, pts, style in (
                ("Same final length as greedy", same, "mark=o, color=gray"),
                ("Different final length", diff, "mark=square*, color=blue!70!black"),
                ("Rescue (greedy wrong, DESPIN right)", rescued, "mark=triangle*, color=red!80!black, mark size=3pt")):
            f.write(f"\\addplot[only marks, {style}] coordinates {{{' '.join(pts)}}};\n\\addlegendentry{{{label}}}\n")
    bins = list(range(80, 600, 40))
    counts = [sum(1 for x in S["_first_pos"] if lo <= x < lo + 40) for lo in bins]
    with open(os.path.join(outdir, "fig_trigger_data.tex"), "w") as f:
        f.write("% AUTO-GENERATED\n\\addplot+[ybar interval, fill=blue!40, draw=blue!70!black, mark=none] coordinates {\n")
        f.write(" ".join(f"({lo},{c})" for lo, c in zip(bins, counts)) + f" ({bins[-1] + 40},0)\n}};\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/logged_run/massive_sweep_all_baselines.json")
    ap.add_argument("--out-dir", default="paper/generated")
    ap.add_argument("--summary", default="results/analysis_summary.json")
    a = ap.parse_args()
    records = load(a.input)
    S = analyse(records)
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.summary) or ".", exist_ok=True)
    write_stats_macros(S, os.path.join(a.out_dir, "stats.tex"))
    write_main_table(S, os.path.join(a.out_dir, "tab_main.tex"))
    write_perproblem(records, os.path.join(a.out_dir, "tab_perproblem.tex"))
    write_figures(S, records, a.out_dir)
    with open(a.summary, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in S.items() if not k.startswith("_")}, f, indent=2)
    m = S["methods"]
    print(f"n={S['n']}  " + "  ".join(f"{k}:{d['correct']}" for k, d in m.items()))
    print("rescues:", [(r["id"], "CAP" if r["capped"] else "ended") for r in S["rescues"]],
          "regressions:", S["regressions"])


if __name__ == "__main__":
    main()

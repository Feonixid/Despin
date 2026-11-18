# DESPIN: rollback and deflation for reasoning loops, plus the audit of its first evaluation

DESPIN watches the step-level final-layer hidden states of a reasoning model for recurrence and
angular-velocity collapse, rolls the KV cache back to the start of the repeated step, applies a short
decaying logit penalty along the displacement vector ("attractor subspace deflation"), and closes
generation early once a boxed answer exists.

**Status (November 2026).** The first version of this project claimed that DESPIN beats greedy decoding,
sampling, repetition penalty and early exit. Re-deriving every number from the raw logs did not support
those claims. This repository now contains the corrected analysis, the original code and logs, and a control
suite that has been **written but not yet run**. Read the paper first: `paper/main.pdf`
(source: `paper/main.tex`, Section 6 lists every withdrawn claim).

Headline numbers from the logged run (60 problems: first 30 MATH-500 + first 30 GSM8K, one run each):
DESPIN 40/60 vs greedy 36/60 (4 gains, 0 losses, exact McNemar p = 0.125); not significantly different from
sampling or repetition penalty; 3 of the 4 gains are runs that hit the 2,048-token cap; counting re-decoded
tokens DESPIN decodes 3.5% more tokens than greedy.

## Layout

```
paper/            main.tex (the bibliography is inside it), main.pdf, generated/ (tables, figure data, macros: do not edit by hand)
scripts/          make_paper_assets.py  - regenerates every number/table/figure-data file from the logs
                  run_controls.py       - control suite (NOT YET RUN; needs a GPU)
src/despin/       detector.py, engine.py, kv_cache.py, evaluator.py (strict, tested)
tests/            test_evaluator.py
data/             problems.json (the 60 problems), logged_run/ (raw logs of the run in the paper)
results/          analysis_summary.json (output of make_paper_assets.py)
legacy/           everything from the first version, kept for provenance (see legacy/README.md)
```

## Quick start

```bash
pip install -e ".[dev]"
pytest                                        # evaluator + statistics tests (no GPU needed)
python scripts/make_paper_assets.py           # regenerate all paper numbers from data/logged_run
cd paper && latexmk -pdf main.tex             # build the paper
```

Run the controls (GPU, about the same cost as the original sweep per configuration):

```bash
python scripts/run_controls.py --configs greedy despin c1_rollback_only c2_random_direction c3_despin_budget c5_closure_only
python scripts/run_controls.py --configs sampling --seeds 5
```

## Reproducing the logged run

The code and engine version that produced `data/logged_run/*.json` are preserved unchanged in
`legacy/logged_run/` (the evaluator there is the original, lenient one; `src/despin/evaluator.py` is the
corrected one and is what new runs use). Default settings of `src/despin/engine.py` reproduce the logged
behaviour; the new options (`deflation_mode`, `budget_includes_pruned`, `seed`) are for the controls.

## Known limitations

One 1.5B model; one run per method; 60 problems; generated texts were not stored in the logged run;
the logged scorer has known false negatives. See the paper, Sections 5-7.

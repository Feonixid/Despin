# legacy/

Code and logs from the first version of this project, kept unchanged so that the audit in `paper/main.pdf`
can be checked. Do not cite numbers from these files without checking them against the paper.

| Path | What it is | Status |
|---|---|---|
| `logged_run/` | The exact engine, detector, KV cache, lenient evaluator and sweep scripts that produced `data/logged_run/*.json` | Provenance. The scripts expect `results_v2/...` paths relative to the working directory. |
| `despin_original/` | All engine and detector versions (v1, v2, v3, tri-regime) | Superseded by `src/despin/`. |
| `results_v2/` | Later result files, including nine intermediate engine variants (`despin_*.json`) | Exploratory runs on small problem subsets. The final run is `massive_sweep_*.json`. |
| `scripts/` | About 49 one-off scripts (benchmarks, plotting, debugging, per-problem tuning for `math500_1`) | Not maintained. Several are cited in the paper's Table 3: `tune_psrd_gating.py` calibrates thresholds on two evaluation items, `generate_case_study_plot.py` contains hard-coded numbers, and `re_score.py` and `test_eval_fix.py` show the evaluator being patched after results were seen. |

The first paper, its figures and the earliest 20-problem results are not part of this repository.

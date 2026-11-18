#!/usr/bin/env python3
"""Control suite for the DESPIN audit.  STATUS: WRITTEN BUT NOT YET EXECUTED.

It re-runs the 60 logged problems under configurations that separate the effect of each component:

  greedy                 baseline, interventions off
  despin                 as in the logged run (rollback + deflation + early closure), kept-length cap
  c1_rollback_only       rollback but NO deflation penalty            (is deflation doing anything?)
  c2_random_direction    rollback + deflation along a RANDOM unit u    (is the direction doing anything?)
  c3_despin_budget       like despin, but max_new_tokens caps ALL decoded tokens (budget-matched)
  c5_closure_only        early closure only, no rollbacks (max_rollbacks=0)
  sampling_seed<k>       temperature 0.6 sampling with seed k          (decoding-noise baseline, C4)

Every generated text is stored, and each run is scored with the strict RobustMathEvaluator
(complete \\boxed{} required). Results are appended to a JSONL file; finished (config, problem, seed)
triples are skipped, so the script can be interrupted and resumed.

Example (from the repository root, after `pip install -e .`):
    python scripts/run_controls.py --configs greedy despin c1_rollback_only c2_random_direction \
        c3_despin_budget c5_closure_only --out results/controls.jsonl
    python scripts/run_controls.py --configs sampling --seeds 5 --out results/controls.jsonl
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from despin.engine import DespinEngineV2
from despin.evaluator import RobustMathEvaluator as Scorer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_NEW = 2048
COMMON = dict(recurrence_threshold=0.90, velocity_collapse_threshold=0.18, consecutive_slow_steps=2,
              warmup_tokens=80, max_rollbacks=2, deflation_gamma=15.0, exploration_steps=6)

# name -> (engine kwargs overriding COMMON, generate kwargs)
CONFIGS = {
    "greedy": ({}, dict(enable_despin=False)),
    "despin": ({}, dict(enable_despin=True)),
    "c1_rollback_only": (dict(deflation_mode="none"), dict(enable_despin=True)),
    "c2_random_direction": (dict(deflation_mode="random", seed=0), dict(enable_despin=True)),
    "c3_despin_budget": (dict(budget_includes_pruned=True), dict(enable_despin=True)),
    "c5_closure_only": (dict(max_rollbacks=0), dict(enable_despin=True)),
    "sampling": ({}, dict(enable_despin=False, base_temperature=0.6)),
}


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                keys.add((r["config"], r["problem_id"], r["seed"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="data/problems.json")
    ap.add_argument("--out", default="results/controls.jsonl")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (used by stochastic configs)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems (smoke test)")
    a = ap.parse_args()

    problems = json.load(open(a.problems, "r", encoding="utf-8"))[: a.limit]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    seen = done_keys(a.out)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")

    for name in a.configs:
        eng_kw, gen_kw = CONFIGS[name]
        engine = DespinEngineV2(model=model, tokenizer=tok, **{**COMMON, **eng_kw})
        seeds = range(a.seeds) if name == "sampling" else [0]
        for seed in seeds:
            for p in problems:
                if (name, p["id"], seed) in seen:
                    continue
                msgs = [{"role": "user", "content":
                         f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {p['question']}"}]
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                torch.manual_seed(seed)
                t0 = time.time()
                res = engine.generate(prompt=prompt, max_new_tokens=MAX_NEW, **gen_kw)
                lat = time.time() - t0
                pred = Scorer.extract_answer(res.text)
                rec = dict(config=name, problem_id=p["id"], seed=seed, correct_strict=Scorer.is_equiv(pred, p["ground_truth"]),
                           pred=pred, kept_tokens=res.final_tokens_count, pruned_tokens=res.pruned_tokens_count,
                           total_decoded=res.final_tokens_count + res.pruned_tokens_count,
                           hit_cap=res.final_tokens_count >= MAX_NEW, latency_s=lat,
                           rollback_history=res.rollback_history, text=res.text)
                with open(a.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{name:22s} seed={seed} {p['id']:11s} strict={'PASS' if rec['correct_strict'] else 'FAIL'} "
                      f"kept={rec['kept_tokens']:4d} pruned={rec['pruned_tokens']:3d}", flush=True)
        del engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

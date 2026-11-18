import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from despin.engine_v2 import DespinEngineV2
from despin.evaluator import MathEvaluator

def run_massive_sweep():
    with open("results_v2/massive_sweep_problems.json", "r", encoding="utf-8") as f:
        problems = json.load(f)

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading {model_id} on GPU...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Initialize Tri-Regime DESPIN Engine
    engine = DespinEngineV2(
        model=model,
        tokenizer=tokenizer,
        recurrence_threshold=0.90,
        velocity_collapse_threshold=0.18,
        consecutive_slow_steps=2,
        warmup_tokens=80,
        max_rollbacks=2,
        deflation_gamma=15.0,
        exploration_steps=6,
    )

    results_file = "results_v2/massive_sweep_live.json"
    records = []

    greedy_correct_total = 0
    despin_correct_total = 0
    greedy_tokens_total = 0
    despin_tokens_total = 0

    greedy_rescued = 0
    pgec_closures = 0

    print(f"\n{'='*75}")
    print(f"STARTING MASSIVE 60-PROBLEM BENCHMARK SWEEP (MATH-500 & GSM8K)")
    print(f"{'='*75}\n", flush=True)

    for i, prob in enumerate(problems):
        pid = prob["id"]
        bench = prob["benchmark"]
        q = prob["question"]
        gt = prob["ground_truth"]

        messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 1. Run Greedy Baseline (enable_despin=False)
        t0 = time.time()
        res_greedy = engine.generate(prompt=prompt, max_new_tokens=2048, enable_despin=False)
        lat_greedy = time.time() - t0
        pred_greedy = MathEvaluator.extract_answer(res_greedy.text)
        corr_greedy = MathEvaluator.is_equiv(pred_greedy, gt)

        # 2. Run Tri-Regime DESPIN (enable_despin=True)
        t1 = time.time()
        res_despin = engine.generate(prompt=prompt, max_new_tokens=2048, enable_despin=True)
        lat_despin = time.time() - t1
        pred_despin = MathEvaluator.extract_answer(res_despin.text)
        corr_despin = MathEvaluator.is_equiv(pred_despin, gt)

        # Update totals
        if corr_greedy:
            greedy_correct_total += 1
        if corr_despin:
            despin_correct_total += 1

        greedy_tokens_total += res_greedy.final_tokens_count
        despin_tokens_total += res_despin.final_tokens_count

        actions = [a["type"] for a in res_despin.rollback_history]
        if "pgec_early_closure" in actions:
            pgec_closures += 1
        if corr_despin and not corr_greedy:
            greedy_rescued += 1

        delta_tok = res_despin.final_tokens_count - res_greedy.final_tokens_count
        status = "SAME"
        if corr_despin and not corr_greedy:
            status = "RESCUED (+)"
        elif not corr_despin and corr_greedy:
            status = "LOST (-)"

        print(
            f"[{i+1:02d}/60] {pid:10s} ({bench:8s}) | "
            f"Greedy: {'PASS' if corr_greedy else 'FAIL':4s} ({res_greedy.final_tokens_count:4d} tok) | "
            f"DESPIN: {'PASS' if corr_despin else 'FAIL':4s} ({res_despin.final_tokens_count:4d} tok) | "
            f"Delta: {delta_tok:+4d} tok | {status:11s} | Actions: {actions}",
            flush=True,
        )

        record = {
            "problem": prob,
            "greedy": {
                "correct": corr_greedy,
                "pred": pred_greedy,
                "tokens": res_greedy.final_tokens_count,
                "latency_s": lat_greedy,
            },
            "despin": {
                "correct": corr_despin,
                "pred": pred_despin,
                "tokens": res_despin.final_tokens_count,
                "pruned": res_despin.pruned_tokens_count,
                "actions": actions,
                "rollback_history": res_despin.rollback_history,
                "latency_s": lat_despin,
            },
        }
        records.append(record)

        # Save live checkpoint after each problem
        with open(results_file, "w", encoding="utf-8") as f:
            n_done = len(records)
            json.dump({
                "completed": n_done,
                "total": len(problems),
                "running_stats": {
                    "greedy_acc": (greedy_correct_total / n_done) * 100,
                    "despin_acc": (despin_correct_total / n_done) * 100,
                    "greedy_avg_tokens": greedy_tokens_total / n_done,
                    "despin_avg_tokens": despin_tokens_total / n_done,
                    "greedy_rescued_count": greedy_rescued,
                    "pgec_closures_count": pgec_closures,
                },
                "records": records,
            }, f, indent=2)

    # Final summary calculation
    n = len(problems)
    g_acc = (greedy_correct_total / n) * 100
    d_acc = (despin_correct_total / n) * 100
    g_tok = greedy_tokens_total / n
    d_tok = despin_tokens_total / n
    g_tes = g_acc / (g_tok / 1000.0)
    d_tes = d_acc / (d_tok / 1000.0)

    print("\n" + "="*75)
    print("MASSIVE SWEEP COMPLETED (N = 60 Problems)")
    print("="*75)
    print(f"Greedy Baseline:    Accuracy = {g_acc:.2f}%, Avg Tokens = {g_tok:.1f}, TES = {g_tes:.2f}")
    print(f"Tri-Regime DESPIN:  Accuracy = {d_acc:.2f}%, Avg Tokens = {d_tok:.1f}, TES = {d_tes:.2f}")
    print(f"Total Trajectories Rescued: {greedy_rescued}")
    print(f"Total PGEC Early Closures:  {pgec_closures}")
    print("="*75 + "\n")

    final_file = "results_v2/massive_sweep_final.json"
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "MATH-500 (30) + GSM8K (30)",
            "model": model_id,
            "total_problems": n,
            "summary": {
                "greedy": {"accuracy": g_acc, "avg_tokens": g_tok, "tes": g_tes},
                "despin": {"accuracy": d_acc, "avg_tokens": d_tok, "tes": d_tes, "rescued": greedy_rescued, "pgec_closures": pgec_closures},
            },
            "records": records,
        }, f, indent=2)
    print(f"Final results permanently archived to {final_file}")

if __name__ == "__main__":
    run_massive_sweep()


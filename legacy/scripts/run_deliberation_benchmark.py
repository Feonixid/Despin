import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from despin.engine_v3 import AdvancedDespinEngine
from despin.evaluator import MathEvaluator

def run_experiment():
    with open('results_v2/benchmark_v2_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    problems = [r['problem'] for r in data['records']]

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    engine = AdvancedDespinEngine(
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

    correct_count = 0
    total_tokens = 0
    total_latency = 0.0
    results = []

    print("=== Running Deliberation-Gated DESPIN Benchmark ===", flush=True)

    for idx, prob in enumerate(problems):
        q = prob['question']
        gt = prob['ground_truth']
        messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        t0 = time.time()
        res = engine.generate(prompt=prompt, max_new_tokens=2048)
        elapsed = time.time() - t0

        pred = MathEvaluator.extract_answer(res.text)
        correct = MathEvaluator.is_equiv(pred, gt)

        if correct:
            correct_count += 1
        total_tokens += res.final_tokens_count
        total_latency += elapsed

        actions = [a['type'] for a in res.rollback_history]
        print(f"[{idx+1:02d}/{len(problems)}] {prob['id']:10s} | Correct: {str(correct):5s} | Tok: {res.final_tokens_count:4d} | Pruned: {res.pruned_tokens_count:3d} | Actions: {actions} | Latency: {elapsed:.2f}s", flush=True)

        results.append({
            "problem_id": prob['id'],
            "correct": correct,
            "pred": pred,
            "tokens": res.final_tokens_count,
            "pruned": res.pruned_tokens_count,
            "actions": actions,
            "latency_s": elapsed,
            "rollback_history": res.rollback_history
        })

    n = len(problems)
    acc = (correct_count / n) * 100
    avg_tokens = total_tokens / n
    avg_lat = total_latency / n
    tes = acc / (avg_tokens / 1000.0)

    print("\n" + "="*60)
    print(f"FINAL ADVANCED DESPIN RESULTS:")
    print(f"Accuracy:     {acc:.2f}% ({correct_count}/{n})")
    print(f"Avg Tokens:   {avg_tokens:.1f}")
    print(f"Avg Latency:  {avg_lat:.2f}s")
    print(f"TES:          {tes:.2f}")
    print("="*60 + "\n")

    out_file = "results_v2/despin_deliberation_gated.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            "accuracy": acc,
            "avg_tokens": avg_tokens,
            "avg_latency": avg_lat,
            "tes": tes,
            "records": results,
        }, f, indent=2)
    print(f"Saved results to {out_file}")

if __name__ == '__main__':
    run_experiment()


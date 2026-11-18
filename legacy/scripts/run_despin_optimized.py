import os
import json
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from despin.engine_v2 import DespinEngineV2
from despin.evaluator import MathEvaluator


def run_despin_sweep():
    # Load existing benchmark records to ensure identical evaluation
    with open('results_v2/benchmark_v2_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    problems = [r['problem'] for r in data['records']]
    print(f"Loaded {len(problems)} evaluation problems from baseline run.\n", flush=True)

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Calibrated DESPIN-v3: Deterministic Deflation (T=0.0 with Subspace Deflation), tau_recur=0.90
    engine = DespinEngineV2(
        model=model,
        tokenizer=tokenizer,
        recurrence_threshold=0.90,
        velocity_collapse_threshold=0.18,
        max_rollbacks=2,
        deflation_gamma=15.0,
        exploration_temp_boost=0.0,  # Purely deterministic orthogonal branching
        exploration_steps=6,
    )

    despin_results = []
    correct_count = 0
    total_tokens = 0
    total_latency = 0.0

    print("=== Running Calibrated DESPIN (PSRD + ASD - Deterministic Deflation) ===", flush=True)

    for idx, prob in enumerate(problems):
        q = prob['question']
        gt = prob['ground_truth']
        messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        t0 = time.time()
        res = engine.generate(
            prompt=prompt,
            max_new_tokens=2048,
            base_temperature=0.0,
            enable_despin=True,
        )
        elapsed = time.time() - t0

        pred = MathEvaluator.extract_answer(res.text)
        correct = MathEvaluator.is_equiv(pred, gt)

        if correct:
            correct_count += 1
        total_tokens += res.total_tokens_generated
        total_latency += elapsed

        despin_results.append({
            "problem_id": prob['id'],
            "correct": correct,
            "pred": pred,
            "tokens": res.total_tokens_generated,
            "pruned": res.pruned_tokens_count,
            "rollbacks": res.rollbacks_count,
            "latency_s": elapsed,
        })

        status = "PASS" if correct else "FAIL"
        print(f"[{idx+1}/{len(problems)}] {prob['id']:12s}: {status} | Pred: {str(pred)[:15]:15s} | GT: {str(gt)[:10]:10s} | Tokens: {res.total_tokens_generated:4d} | Rollbacks: {res.rollbacks_count:2d}", flush=True)

    acc = correct_count / len(problems) * 100.0
    avg_tokens = total_tokens / len(problems)
    avg_lat = total_latency / len(problems)
    tes = acc / (avg_tokens / 1000.0)

    print("\n" + "=" * 65, flush=True)
    print(f"DESPIN Calibrated: Acc = {acc:.2f}% | Avg Tokens = {avg_tokens:.1f} | TES = {tes:.2f}", flush=True)
    print("=" * 65, flush=True)

    # Save results
    with open('results_v2/despin_calibrated.json', 'w', encoding='utf-8') as f:
        json.dump({
            "accuracy": acc,
            "avg_tokens": avg_tokens,
            "avg_latency": avg_lat,
            "tes": tes,
            "records": despin_results,
        }, f, indent=2)

if __name__ == '__main__':
    run_despin_sweep()


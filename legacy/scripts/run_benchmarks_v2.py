import os
import json
import time
from typing import List, Dict, Any
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from despin.engine_v2 import DespinEngineV2
from despin.evaluator import MathEvaluator


def load_math500_sample(n: int = 10) -> List[Dict[str, Any]]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = []
    for item in ds:
        problems.append({
            "id": f"math500_{len(problems)}",
            "benchmark": "MATH-500",
            "subject": item.get("subject", "General"),
            "question": item["problem"],
            "ground_truth": item["answer"],
        })
        if len(problems) >= n:
            break
    return problems


def load_gsm8k_sample(n: int = 10) -> List[Dict[str, Any]]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for i, item in enumerate(ds):
        gt = item["answer"].split("####")[-1].strip()
        problems.append({
            "id": f"gsm8k_{i}",
            "benchmark": "GSM8K",
            "subject": "Arithmetic",
            "question": item["question"],
            "ground_truth": gt,
        })
        if len(problems) >= n:
            break
    return problems


def run_benchmark_suite_v2(
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    num_math: int = 10,
    num_gsm: int = 10,
    output_dir: str = "results_v2",
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Initializing Benchmark Suite V2 with {model_id} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    engine = DespinEngineV2(model=model, tokenizer=tokenizer)

    math_problems = load_math500_sample(num_math)
    gsm_problems = load_gsm8k_sample(num_gsm)
    all_problems = math_problems + gsm_problems
    print(f"Loaded {len(all_problems)} total evaluation problems ({len(math_problems)} MATH-500, {len(gsm_problems)} GSM8K).\n", flush=True)

    methods = [
        {"name": "Greedy (T=0.0)", "type": "greedy"},
        {"name": "Sampling (T=0.6)", "type": "sampling"},
        {"name": "Best-of-2 (T=0.6)", "type": "bon2"},
        {"name": "Passive Early-Exit (Wei et al. 2025)", "type": "passive_ee"},
        {"name": "DESPIN-v2 (PSRD + ASD)", "type": "despin_v2"},
    ]

    all_records = []
    summary_by_method = {m["name"]: {"correct": 0, "total": 0, "tokens": 0, "latency": 0.0, "rescued": 0} for m in methods}

    for idx, prob in enumerate(all_problems):
        q = prob["question"]
        gt = prob["ground_truth"]
        messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        print(f"[{idx+1}/{len(all_problems)}] Problem {prob['id']} ({prob['benchmark']} - {prob['subject']})", flush=True)

        prob_results = {}
        for m in methods:
            m_name = m["name"]
            m_type = m["type"]
            t0 = time.time()

            if m_type == "greedy":
                res = engine.generate(prompt=prompt, max_new_tokens=2048, base_temperature=0.0, enable_despin=False)
                pred = MathEvaluator.extract_answer(res.text)
                correct = MathEvaluator.is_equiv(pred, gt)
                tot_tokens = res.total_tokens_generated
                rollbacks = 0

            elif m_type == "sampling":
                res = engine.generate(prompt=prompt, max_new_tokens=2048, base_temperature=0.6, top_p=0.95, enable_despin=False)
                pred = MathEvaluator.extract_answer(res.text)
                correct = MathEvaluator.is_equiv(pred, gt)
                tot_tokens = res.total_tokens_generated
                rollbacks = 0

            elif m_type == "bon2":
                # Best-of-2 Sampling: generate 2 samples, take agreement or higher confidence
                res1 = engine.generate(prompt=prompt, max_new_tokens=2048, base_temperature=0.6, top_p=0.95, enable_despin=False)
                res2 = engine.generate(prompt=prompt, max_new_tokens=2048, base_temperature=0.6, top_p=0.95, enable_despin=False)
                pred1 = MathEvaluator.extract_answer(res1.text)
                pred2 = MathEvaluator.extract_answer(res2.text)
                # If either is correct, count as hit (oracle / best verification)
                correct1 = MathEvaluator.is_equiv(pred1, gt)
                correct2 = MathEvaluator.is_equiv(pred2, gt)
                correct = correct1 or correct2
                pred = pred1 if correct1 else pred2
                tot_tokens = res1.total_tokens_generated + res2.total_tokens_generated
                rollbacks = 0

            elif m_type == "passive_ee":
                res = engine.generate(prompt=prompt, max_new_tokens=2048, base_temperature=0.0, enable_despin=False, passive_early_exit=True)
                pred = MathEvaluator.extract_answer(res.text)
                correct = MathEvaluator.is_equiv(pred, gt)
                tot_tokens = res.total_tokens_generated
                rollbacks = res.rollbacks_count

            elif m_type == "despin_v2":
                res = engine.generate(prompt=prompt, max_new_tokens=2048, base_temperature=0.0, enable_despin=True)
                pred = MathEvaluator.extract_answer(res.text)
                correct = MathEvaluator.is_equiv(pred, gt)
                tot_tokens = res.total_tokens_generated
                rollbacks = res.rollbacks_count

            elapsed = time.time() - t0

            summary_by_method[m_name]["total"] += 1
            if correct:
                summary_by_method[m_name]["correct"] += 1
            summary_by_method[m_name]["tokens"] += tot_tokens
            summary_by_method[m_name]["latency"] += elapsed

            prob_results[m_name] = {
                "correct": correct,
                "pred": pred,
                "tokens": tot_tokens,
                "rollbacks": rollbacks,
                "latency_s": elapsed,
            }

            status_str = "PASS" if correct else "FAIL"
            print(f"  > {m_name:38s}: {status_str} | Pred: {str(pred)[:15]:15s} | Tokens: {tot_tokens:4d} | Rollbacks: {rollbacks:2d}", flush=True)

        # Check trajectory rescue vs Greedy and Passive Early-Exit
        if not prob_results["Passive Early-Exit (Wei et al. 2025)"]["correct"] and prob_results["DESPIN-v2 (PSRD + ASD)"]["correct"]:
            summary_by_method["DESPIN-v2 (PSRD + ASD)"]["rescued"] += 1
            print("  *** TRAJECTORY RESCUED by DESPIN-v2! ***", flush=True)

        all_records.append({"problem": prob, "results": prob_results})

    # Compile Summary Table
    summary_table = []
    for m in methods:
        m_name = m["name"]
        stats = summary_by_method[m_name]
        tot = stats["total"]
        acc = (stats["correct"] / tot * 100.0) if tot else 0.0
        avg_tokens = (stats["tokens"] / tot) if tot else 0.0
        avg_latency = (stats["latency"] / tot) if tot else 0.0
        tes = (acc / (avg_tokens / 1000.0)) if avg_tokens else 0.0

        summary_table.append({
            "Method": m_name,
            "Accuracy (%)": round(acc, 2),
            "Correct": stats["correct"],
            "Total": tot,
            "Avg Tokens": round(avg_tokens, 1),
            "Avg Latency (s)": round(avg_latency, 2),
            "Token Efficiency Score (TES)": round(tes, 2),
            "Rescued Trajectories": stats.get("rescued", 0),
        })

    results_payload = {
        "model": model_id,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_new_tokens": 2048,
        "summary": summary_table,
        "records": all_records,
    }

    out_file = os.path.join(output_dir, "benchmark_v2_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}", flush=True)

    print("\n" + "=" * 85, flush=True)
    print(f"{'Method':40s} | {'Accuracy':10s} | {'Avg Tokens':11s} | {'TES':8s} | {'Rescued':8s}", flush=True)
    print("-" * 85, flush=True)
    for row in summary_table:
        print(f"{row['Method']:40s} | {row['Accuracy (%)']:9.2f}% | {row['Avg Tokens']:11.1f} | {row['Token Efficiency Score (TES)']:8.2f} | {row['Rescued Trajectories']:8d}", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    run_benchmark_suite_v2(num_math=10, num_gsm=10)


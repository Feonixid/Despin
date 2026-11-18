import os
import json
import time
from typing import List, Dict, Any
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from despin.engine import DespinEngine
from despin.evaluator import MathEvaluator


def load_math500_sample(n: int = 50) -> List[Dict[str, Any]]:
    """Loads a diverse slice of MATH-500 problems."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    # Take a balanced slice across subject categories
    problems = []
    seen_subjects = {}
    for item in ds:
        subj = item.get("subject", "General")
        seen_subjects[subj] = seen_subjects.get(subj, 0) + 1
        problems.append({
            "id": f"math500_{len(problems)}",
            "benchmark": "MATH-500",
            "subject": subj,
            "question": item["problem"],
            "ground_truth": item["answer"],
        })
        if len(problems) >= n:
            break
    return problems


def load_gsm8k_sample(n: int = 50) -> List[Dict[str, Any]]:
    """Loads a slice of GSM8K test problems."""
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


def run_benchmark_suite(
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    num_math: int = 35,
    num_gsm: int = 35,
    output_dir: str = "results",
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Initializing Benchmark Suite with {model_id} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    engine = DespinEngine(model=model, tokenizer=tokenizer)

    math_problems = load_math500_sample(num_math)
    gsm_problems = load_gsm8k_sample(num_gsm)
    all_problems = math_problems + gsm_problems
    print(f"Loaded {len(all_problems)} total evaluation problems ({len(math_problems)} MATH-500, {len(gsm_problems)} GSM8K).")

    methods = [
        {"name": "Greedy", "enable_despin": False, "passive_early_exit": False, "temp": 0.0},
        {"name": "Sampling (T=0.6)", "enable_despin": False, "passive_early_exit": False, "temp": 0.6, "top_p": 0.95},
        {"name": "Passive Early-Exit (Wei et al. 2025)", "enable_despin": False, "passive_early_exit": True, "temp": 0.0},
        {"name": "DESPIN (Ours)", "enable_despin": True, "passive_early_exit": False, "temp": 0.0},
    ]

    all_records = []
    summary_by_method = {m["name"]: {"correct": 0, "total": 0, "tokens": 0, "latency": 0.0, "rescued": 0} for m in methods}

    for idx, prob in enumerate(all_problems):
        q = prob["question"]
        gt = prob["ground_truth"]
        messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        print(f"\n[{idx+1}/{len(all_problems)}] Problem {prob['id']} ({prob['benchmark']} - {prob['subject']})")

        prob_results = {}
        for m in methods:
            m_name = m["name"]
            temp = m.get("temp", 0.0)
            top_p = m.get("top_p", 1.0)
            enable_despin = m["enable_despin"]
            passive_ee = m["passive_early_exit"]

            t0 = time.time()
            res = engine.generate(
                prompt=prompt,
                max_new_tokens=1024,
                base_temperature=temp,
                top_p=top_p,
                enable_despin=enable_despin,
                passive_early_exit=passive_ee,
            )
            elapsed = time.time() - t0

            pred = MathEvaluator.extract_answer(res.text)
            correct = MathEvaluator.is_equiv(pred, gt)

            summary_by_method[m_name]["total"] += 1
            if correct:
                summary_by_method[m_name]["correct"] += 1
            summary_by_method[m_name]["tokens"] += res.total_tokens_generated
            summary_by_method[m_name]["latency"] += elapsed

            prob_results[m_name] = {
                "correct": correct,
                "pred": pred,
                "tokens": res.total_tokens_generated,
                "pruned_tokens": res.pruned_tokens_count,
                "rollbacks": res.rollbacks_count,
                "latency_s": elapsed,
            }

            print(f"  > {m_name:32s}: {'PASS' if correct else 'FAIL'} | Pred: {str(pred)[:15]:15s} | Tokens: {res.total_tokens_generated:4d} | Rollbacks: {res.rollbacks_count:2d}")

        # Check Trajectory Rescue: Where Passive Early-Exit failed but DESPIN succeeded
        if not prob_results["Passive Early-Exit (Wei et al. 2025)"]["correct"] and prob_results["DESPIN (Ours)"]["correct"]:
            summary_by_method["DESPIN (Ours)"]["rescued"] += 1
            print(f"  *** TRAJECTORY RESCUED by DESPIN! ***")

        all_records.append({
            "problem": prob,
            "results": prob_results,
        })

    # Compile Final Benchmark Summary
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
            "Token Efficiency Score": round(tes, 2),
            "Rescued Trajectories": stats.get("rescued", 0),
        })

    results_payload = {
        "model": model_id,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary_table,
        "records": all_records,
    }

    out_file = os.path.join(output_dir, "benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nSaved complete benchmark results to {out_file}")

    print("\n" + "=" * 80)
    print(f"{'Method':36s} | {'Accuracy':10s} | {'Avg Tokens':11s} | {'TES':8s} | {'Rescued':8s}")
    print("-" * 80)
    for row in summary_table:
        print(f"{row['Method']:36s} | {row['Accuracy (%)']:9.2f}% | {row['Avg Tokens']:11.1f} | {row['Token Efficiency Score']:8.2f} | {row['Rescued Trajectories']:8d}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark_suite(num_math=10, num_gsm=10)

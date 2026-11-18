import os
import json
import time
from typing import List, Dict, Any
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from despin.engine import DespinEngine
from despin.evaluator import MathEvaluator


def run_ablation_study(
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    num_problems: int = 15,
    output_dir: str = "results",
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Initializing DESPIN Ablation Study with {model_id} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Load challenging MATH problems
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = []
    for item in ds:
        problems.append({
            "id": f"ablation_math_{len(problems)}",
            "question": item["problem"],
            "ground_truth": item["answer"],
        })
        if len(problems) >= num_problems:
            break

    # Ablation configurations
    configurations = [
        {
            "name": "DESPIN (Full)",
            "lambda_logit": 8.0,
            "temp_boost": 0.5,
            "ngram_thresh": 0.40,
        },
        {
            "name": "w/o Directional Penalty (lambda=0)",
            "lambda_logit": 0.0,
            "temp_boost": 0.5,
            "ngram_thresh": 0.40,
        },
        {
            "name": "w/o Exploration Boost (temp_boost=0)",
            "lambda_logit": 8.0,
            "temp_boost": 0.0,
            "ngram_thresh": 0.40,
        },
        {
            "name": "High Sensitivity (thresh=0.30)",
            "lambda_logit": 8.0,
            "temp_boost": 0.5,
            "ngram_thresh": 0.30,
        },
        {
            "name": "Low Sensitivity (thresh=0.60)",
            "lambda_logit": 8.0,
            "temp_boost": 0.5,
            "ngram_thresh": 0.60,
        },
    ]

    ablation_results = {cfg["name"]: {"correct": 0, "total": 0, "tokens": 0, "rollbacks": 0} for cfg in configurations}

    for idx, prob in enumerate(problems):
        q = prob["question"]
        gt = prob["ground_truth"]
        messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        print(f"\n[Ablation {idx+1}/{len(problems)}] Problem {prob['id']}")

        for cfg in configurations:
            cfg_name = cfg["name"]
            engine = DespinEngine(
                model=model,
                tokenizer=tokenizer,
                ngram_repeat_threshold=cfg["ngram_thresh"],
                logit_penalty_lambda=cfg["lambda_logit"],
                exploration_temp_boost=cfg["temp_boost"],
            )

            res = engine.generate(prompt=prompt, max_new_tokens=1024, enable_despin=True)
            pred = MathEvaluator.extract_answer(res.text)
            correct = MathEvaluator.is_equiv(pred, gt)

            ablation_results[cfg_name]["total"] += 1
            if correct:
                ablation_results[cfg_name]["correct"] += 1
            ablation_results[cfg_name]["tokens"] += res.total_tokens_generated
            ablation_results[cfg_name]["rollbacks"] += res.rollbacks_count

            print(f"  > {cfg_name:38s}: {'PASS' if correct else 'FAIL'} | Tokens: {res.total_tokens_generated:4d} | Rollbacks: {res.rollbacks_count:2d}")

    # Summary Table
    summary = []
    for cfg in configurations:
        cfg_name = cfg["name"]
        stats = ablation_results[cfg_name]
        tot = stats["total"]
        acc = (stats["correct"] / tot * 100.0) if tot else 0.0
        avg_tokens = (stats["tokens"] / tot) if tot else 0.0
        avg_rollbacks = (stats["rollbacks"] / tot) if tot else 0.0

        summary.append({
            "Configuration": cfg_name,
            "Accuracy (%)": round(acc, 2),
            "Avg Tokens": round(avg_tokens, 1),
            "Avg Rollbacks": round(avg_rollbacks, 2),
        })

    out_file = os.path.join(output_dir, "ablation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw": ablation_results}, f, indent=2)
    print(f"\nSaved ablation results to {out_file}")

    print("\n" + "=" * 75)
    print(f"{'Configuration':40s} | {'Accuracy':10s} | {'Avg Tokens':11s} | {'Rollbacks':9s}")
    print("-" * 75)
    for row in summary:
        print(f"{row['Configuration']:40s} | {row['Accuracy (%)']:9.2f}% | {row['Avg Tokens']:11.1f} | {row['Avg Rollbacks']:9.2f}")
    print("=" * 75)


if __name__ == "__main__":
    run_ablation_study(num_problems=10)


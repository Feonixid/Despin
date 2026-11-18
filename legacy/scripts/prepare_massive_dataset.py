import json
from datasets import load_dataset

def prepare_dataset(n_math=30, n_gsm=30):
    print(f"Loading {n_math} MATH-500 and {n_gsm} GSM8K problems...")
    ds_math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds_gsm = load_dataset("openai/gsm8k", "main", split="test")

    problems = []

    # MATH-500 selection: diverse subjects and levels
    math_count = 0
    for item in ds_math:
        problems.append({
            "id": f"math500_{math_count}",
            "benchmark": "MATH-500",
            "subject": item.get("subject", "General"),
            "level": item.get("level", 1),
            "question": item["problem"],
            "ground_truth": item["answer"],
        })
        math_count += 1
        if math_count >= n_math:
            break

    # GSM8K selection
    gsm_count = 0
    for item in ds_gsm:
        gt = item["answer"].split("####")[-1].strip()
        problems.append({
            "id": f"gsm8k_{gsm_count}",
            "benchmark": "GSM8K",
            "subject": "Arithmetic",
            "level": 1,
            "question": item["question"],
            "ground_truth": gt,
        })
        gsm_count += 1
        if gsm_count >= n_gsm:
            break

    out_file = "results_v2/massive_sweep_problems.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2)

    print(f"Prepared {len(problems)} total problems ({math_count} MATH-500, {gsm_count} GSM8K) saved to {out_file}")

    # Inspect subject distribution for MATH-500
    from collections import Counter
    subjects = Counter(p["subject"] for p in problems if p["benchmark"] == "MATH-500")
    print("MATH-500 subjects:", dict(subjects))

if __name__ == "__main__":
    prepare_dataset(30, 30)


import json

with open('results_v2/benchmark_v2_results.json', 'r', encoding='utf-8') as f:
    baseline_data = json.load(f)

with open('results_v2/despin_calibrated.json', 'r', encoding='utf-8') as f:
    despin_data = json.load(f)

from despin.evaluator import MathEvaluator

methods = ["Greedy (T=0.0)", "Sampling (T=0.6)", "Best-of-2 (T=0.6)", "Passive Early-Exit (Wei et al. 2025)"]
stats = {m: {"correct": 0, "total": 0, "tokens": 0, "latency": 0.0} for m in methods}

for r in baseline_data['records']:
    gt = r['problem']['ground_truth']
    for m in methods:
        pred = r['results'][m]['pred']
        corr = MathEvaluator.is_equiv(pred, gt)
        stats[m]['total'] += 1
        if corr:
            stats[m]['correct'] += 1
        stats[m]['tokens'] += r['results'][m]['tokens']
        stats[m]['latency'] += r['results'][m]['latency_s']

# Add Calibrated DESPIN
stats["DESPIN-v2 (Calibrated)"] = {
    "correct": int(despin_data['accuracy'] * len(despin_data['records']) / 100.0),
    "total": len(despin_data['records']),
    "tokens": int(despin_data['avg_tokens'] * len(despin_data['records'])),
    "latency": despin_data['avg_latency'] * len(despin_data['records']),
}

final_summary = []
print("=" * 90)
print(f"{'Method':40s} | {'Accuracy':10s} | {'Avg Tokens':11s} | {'Avg Latency':12s} | {'TES':8s}")
print("-" * 90)

all_m = list(methods) + ["DESPIN-v2 (Calibrated)"]
for m in all_m:
    tot = stats[m]['total']
    acc = stats[m]['correct'] / tot * 100.0
    tokens = stats[m]['tokens'] / tot
    lat = stats[m]['latency'] / tot
    tes = acc / (tokens / 1000.0)
    final_summary.append({
        "Method": m,
        "Accuracy (%)": round(acc, 2),
        "Avg Tokens": round(tokens, 1),
        "Avg Latency (s)": round(lat, 2),
        "Token Efficiency Score (TES)": round(tes, 2),
    })
    print(f"{m:40s} | {acc:9.2f}% | {tokens:11.1f} | {lat:10.2f}s | {tes:8.2f}")
print("=" * 90)

with open('results_v2/final_rigorous_comparison.json', 'w', encoding='utf-8') as f:
    json.dump({"summary": final_summary}, f, indent=2)


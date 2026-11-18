import json
from despin.evaluator import MathEvaluator

with open('results_v2/benchmark_v2_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

methods = ["Greedy (T=0.0)", "Sampling (T=0.6)", "Best-of-2 (T=0.6)", "Passive Early-Exit (Wei et al. 2025)", "DESPIN-v2 (PSRD + ASD)"]
scores = {m: {"correct": 0, "total": 0, "tokens": 0} for m in methods}

for r in data['records']:
    gt = r['problem']['ground_truth']
    for m in methods:
        pred = r['results'][m]['pred']
        corr = MathEvaluator.is_equiv(pred, gt)
        r['results'][m]['correct'] = corr
        scores[m]['total'] += 1
        if corr:
            scores[m]['correct'] += 1
        scores[m]['tokens'] += r['results'][m]['tokens']

print("=" * 80)
print(f"{'Method':40s} | {'Accuracy':10s} | {'Avg Tokens':11s} | {'TES':8s}")
print("-" * 80)
for m in methods:
    acc = scores[m]['correct'] / scores[m]['total'] * 100.0
    avg_tok = scores[m]['tokens'] / scores[m]['total']
    tes = acc / (avg_tok / 1000.0)
    print(f"{m:40s} | {acc:9.2f}% | {avg_tok:11.1f} | {tes:8.2f}")
print("=" * 80)


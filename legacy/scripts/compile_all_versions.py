import json
from despin.evaluator import MathEvaluator

def compile_all():
    with open('results_v2/benchmark_v2_results.json', 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)

    with open('results_v2/despin_hyper_efficient.json', 'r', encoding='utf-8') as f:
        hyper_data = json.load(f)

    with open('results_v2/despin_calibrated.json', 'r', encoding='utf-8') as f:
        v2_data = json.load(f)

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

    stats["DESPIN-v2 (Rollback Only)"] = {
        "correct": int(v2_data['accuracy'] * len(v2_data['records']) / 100.0),
        "total": len(v2_data['records']),
        "tokens": int(v2_data['avg_tokens'] * len(v2_data['records'])),
        "latency": v2_data['avg_latency'] * len(v2_data['records']),
    }

    stats["DESPIN (Hyper-Efficient GS+PGEC)"] = {
        "correct": int(hyper_data['accuracy'] * len(hyper_data['records']) / 100.0),
        "total": len(hyper_data['records']),
        "tokens": int(hyper_data['avg_tokens'] * len(hyper_data['records'])),
        "latency": hyper_data['avg_latency'] * len(hyper_data['records']),
    }

    print("=" * 100)
    print(f"{'Method':45s} | {'Accuracy':10s} | {'Avg Tokens':11s} | {'Avg Latency':12s} | {'TES':8s}")
    print("-" * 100)

    summary_list = []
    all_m = list(methods) + ["DESPIN-v2 (Rollback Only)", "DESPIN (Hyper-Efficient GS+PGEC)"]
    for m in all_m:
        tot = stats[m]['total']
        acc = stats[m]['correct'] / tot * 100.0
        tokens = stats[m]['tokens'] / tot
        lat = stats[m]['latency'] / tot
        tes = acc / (tokens / 1000.0)
        summary_list.append({
            "Method": m,
            "Accuracy (%)": round(acc, 2),
            "Avg Tokens": round(tokens, 1),
            "Avg Latency (s)": round(lat, 2),
            "Token Efficiency Score (TES)": round(tes, 2),
        })
        print(f"{m:45s} | {acc:9.2f}% | {tokens:11.1f} | {lat:10.2f}s | {tes:8.2f}")
    print("=" * 100)

    with open('results_v2/hyper_efficient_summary.json', 'w', encoding='utf-8') as f:
        json.dump({"summary": summary_list}, f, indent=2)

if __name__ == '__main__':
    compile_all()


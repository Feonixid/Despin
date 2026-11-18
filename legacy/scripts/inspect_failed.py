import json

with open('results_v2/benchmark_v2_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for r in data['records']:
    pid = r['problem']['id']
    bench = r['problem']['benchmark']
    res = r['results']
    g_corr = res['Greedy (T=0.0)']['correct']
    d_corr = res['DESPIN-v2 (PSRD + ASD)']['correct']
    bon_corr = res['Best-of-2 (T=0.6)']['correct']
    samp_corr = res['Sampling (T=0.6)']['correct']

    if not g_corr:
        print(f"Failed by Greedy: {pid} ({bench})")
        print(f"  GT: {r['problem']['ground_truth']}")
        print(f"  Greedy: {res['Greedy (T=0.0)']['pred']}")
        print(f"  Sampling: {res['Sampling (T=0.6)']['pred']} (correct={samp_corr})")
        print(f"  Best-of-2: {res['Best-of-2 (T=0.6)']['pred']} (correct={bon_corr})")
        print(f"  DESPIN: {res['DESPIN-v2 (PSRD + ASD)']['pred']} (rollbacks={res['DESPIN-v2 (PSRD + ASD)']['rollbacks']})")
        print()


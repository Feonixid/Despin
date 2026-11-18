import json

with open('results_v2/benchmark_v2_results.json', 'r') as f:
    d = json.load(f)

for r in d['records']:
    if 'gsm' in r['problem']['id']:
        res = r['results']['DESPIN-v2 (PSRD + ASD)']
        print(f"{r['problem']['id']}: tokens={res['tokens']}, rollbacks={res['rollbacks']}")


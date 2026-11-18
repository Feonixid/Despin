import json

with open('results_v2/benchmark_v2_results.json', 'r', encoding='utf-8') as f:
    base = json.load(f)

with open('results_v2/despin_gram_schmidt.json', 'r', encoding='utf-8') as f:
    gs = json.load(f)

print(f"{'Problem ID':12s} | {'Greedy Tok':10s} | {'GS Tok':10s} | {'Delta Tok':10s} | {'Greedy Lat':10s} | {'GS Lat':10s} | {'Rollbacks/Actions':30s}")
print("-" * 105)

for i in range(len(base['records'])):
    pid = base['records'][i]['problem']['id']
    g_tok = base['records'][i]['results']['Greedy (T=0.0)']['tokens']
    g_lat = base['records'][i]['results']['Greedy (T=0.0)']['latency_s']
    gs_tok = gs['records'][i]['tokens']
    gs_lat = gs['records'][i]['latency_s']
    diff_tok = gs_tok - g_tok
    actions = str(gs['records'][i]['actions'])
    print(f"{pid:12s} | {g_tok:10d} | {gs_tok:10d} | {diff_tok:+10d} | {g_lat:9.2f}s | {gs_lat:9.2f}s | {actions:30s}")


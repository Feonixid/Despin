import json

with open('results_v2/benchmark_v2_results.json') as f:
    b2 = json.load(f)
with open('results_v2/despin_production_final.json') as f:
    prod = json.load(f)

prod_by_id = {r['problem_id']: r for r in prod['records']}

print(f"{'Problem':10s} | {'Greedy':12s} | {'Sampling':12s} | {'DESPIN':12s} | {'Delta Tok (D - G)':17s} | Notes")
print("-" * 85)

for r in b2['records']:
    pid = r['problem']['id']
    g = r['results']['Greedy (T=0.0)']
    s = r['results']['Sampling (T=0.6)']
    d = prod_by_id[pid]
    
    g_str = f"{str(g['correct']):5s} ({g['tokens']})"
    s_str = f"{str(s['correct']):5s} ({s['tokens']})"
    d_str = f"{str(d['correct']):5s} ({d['tokens']})"
    diff = d['tokens'] - g['tokens']
    
    note = []
    if d['correct'] and not g['correct']:
        note.append("Rescued vs Greedy")
    if not d['correct'] and g['correct']:
        note.append("LOST vs Greedy")
    if d['correct'] and not s['correct']:
        note.append("Won vs Sampling")
    if not d['correct'] and s['correct']:
        note.append("Lost vs Sampling")
    if d['actions']:
        note.append(f"{','.join(d['actions'])}")
        
    print(f"{pid:10s} | {g_str:12s} | {s_str:12s} | {d_str:12s} | {diff:+6d} tokens   | {'; '.join(note)}")


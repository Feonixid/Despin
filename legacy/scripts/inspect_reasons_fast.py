import json

with open('results_v2/despin_principled_fast.json', 'r') as f:
    d = json.load(f)

for r in d['records']:
    if r['actions']:
        print(f"{r['problem_id']:12s} | Correct: {str(r['correct']):5s} | Tok: {r['tokens']:4d} | Actions: {r['actions']}")


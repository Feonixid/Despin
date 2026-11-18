import json

with open('results_v2/despin_calibrated.json', 'r') as f:
    calib = json.load(f)

r_calib = next(r for r in calib['records'] if r['problem_id'] == 'math500_1')
print("Calibrated math500_1:", r_calib)

with open('results_v2/despin_hyper_efficient.json', 'r') as f:
    hyper = json.load(f)

r_hyper = next(r for r in hyper['records'] if r['problem_id'] == 'math500_1')
print("Hyper math500_1:", r_hyper)


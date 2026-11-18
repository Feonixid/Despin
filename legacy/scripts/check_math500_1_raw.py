import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from despin.engine_v2 import DespinEngineV2

with open('results_v2/benchmark_v2_results.json', 'r') as f:
    data = json.load(f)

prob = next(r['problem'] for r in data['records'] if r['problem']['id'] == 'math500_1')
print("Question snippet:", prob['question'][:100])
print("Ground truth:", prob['ground_truth'])


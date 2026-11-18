import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from despin.evaluator import MathEvaluator
from test_hyper_efficient import HyperEfficientEngine

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")

with open('results_v2/benchmark_v2_results.json', 'r') as f:
    import json
    data = json.load(f)

prob = next(r['problem'] for r in data['records'] if r['problem']['id'] == 'math500_1')
q = prob['question']
gt = prob['ground_truth']
messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Test with recurrence_threshold = 0.90, warmup_tokens = 100
engine = HyperEfficientEngine(
    model=model,
    tokenizer=tokenizer,
    recurrence_threshold=0.90,
    velocity_collapse_threshold=0.18,
    consecutive_slow_steps=2,
    warmup_tokens=100,
    deflation_gamma=15.0,
    exploration_steps=6,
)

res = engine.generate(prompt=prompt, max_new_tokens=2048, enable_pgec=True)
pred = MathEvaluator.extract_answer(res.text)
correct = MathEvaluator.is_equiv(pred, gt)

print(f"math500_1 Result: Correct={correct} | Pred={pred} | GT={gt} | Tokens={res.total_tokens_generated} | Pruned={res.pruned_tokens_count}")


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from despin.engine_v2 import DespinEngineV2
from despin.evaluator import MathEvaluator
import json

with open('results_v2/benchmark_v2_results.json', 'r') as f:
    data = json.load(f)

prob = next(r['problem'] for r in data['records'] if r['problem']['id'] == 'math500_1')
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", dtype=torch.bfloat16, device_map="cuda")

engine = DespinEngineV2(
    model=model,
    tokenizer=tokenizer,
    recurrence_threshold=0.90,
    velocity_collapse_threshold=0.18,
    max_rollbacks=2,
    deflation_gamma=15.0,
    exploration_temp_boost=0.0,
    exploration_steps=6,
    enable_pgec=True,
)

q = prob['question']
messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

res = engine.generate(prompt=prompt, max_new_tokens=2048)
pred = MathEvaluator.extract_answer(res.text)
print("Result with current engine_v2:", pred)
print("Correct:", MathEvaluator.is_equiv(pred, prob['ground_truth']))
print("Rollback history:", res.rollback_history)


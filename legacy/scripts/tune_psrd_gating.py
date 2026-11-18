import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from despin.detector_v2 import PhaseSpaceRecurrenceDetector

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")

# Let's test on gsm8k_0 (clean solution) and math500_1 (looping problem)
with open('results_v2/benchmark_v2_results.json', 'r', encoding='utf-8') as f:
    import json
    data = json.load(f)

gsm_prob = next(r['problem'] for r in data['records'] if r['problem']['id'] == 'gsm8k_0')
math_prob = next(r['problem'] for r in data['records'] if r['problem']['id'] == 'math500_1')

def trace_problem(prob):
    q = prob['question']
    messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=600, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)
    gen_tokens = out.sequences[0, input_ids.shape[1]:]

    detector = PhaseSpaceRecurrenceDetector(tokenizer)
    for idx, tok in enumerate(gen_tokens):
        hs = out.hidden_states[idx][-1][0, -1, :]
        detector.step(tok.item(), hs)
    return detector.steps

print("Capturing traces...")
gsm_steps = trace_problem(gsm_prob)
math_steps = trace_problem(math_prob)

print(f"\nGSM8K-0: {len(gsm_steps)} steps")
print(f"MATH500-1: {len(math_steps)} steps")

def eval_condition(name, cond_fn, steps):
    triggers = []
    n = len(steps)
    for k in range(3, n):
        # find best_sim among j <= k-2
        best_sim = -1
        best_j = -1
        for j in range(k - 2):
            sim = F.cosine_similarity(steps[k].hidden_state.unsqueeze(0), steps[j].hidden_state.unsqueeze(0), dim=-1).item()
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if cond_fn(steps[k], steps[best_j], best_sim, k - best_j):
            triggers.append((k, best_j, best_sim, steps[k].velocity))
    return triggers

conditions = {
    "Old (sim >= 0.90)": lambda sk, sj, sim, period: sim >= 0.90,
    "Velocity-Gated (sim >= 0.90 and vel < 0.40)": lambda sk, sj, sim, period: sim >= 0.90 and sk.velocity < 0.40,
    "Tight Recurrence (sim >= 0.93)": lambda sk, sj, sim, period: sim >= 0.93,
    "Combined (sim >= 0.92 or (sim >= 0.88 and vel < 0.25))": lambda sk, sj, sim, period: sim >= 0.92 or (sim >= 0.88 and sk.velocity < 0.25),
}

for name, fn in conditions.items():
    gsm_trig = eval_condition(name, fn, gsm_steps)
    math_trig = eval_condition(name, fn, math_steps)
    print(f"\nCondition: {name}")
    print(f"  GSM8K-0 (Clean) False Positives: {len(gsm_trig)} triggers: {gsm_trig[:3]}")
    print(f"  MATH500-1 (Loop) True Positives: {len(math_trig)} triggers: {math_trig[:3]}")


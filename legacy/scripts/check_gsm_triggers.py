import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from despin.detector_v2 import PhaseSpaceRecurrenceDetector

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")

with open('results_v2/benchmark_v2_results.json', 'r') as f:
    import json
    data = json.load(f)

for pid in ['gsm8k_0', 'gsm8k_6']:
    prob = next(r['problem'] for r in data['records'] if r['problem']['id'] == pid)
    messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {prob['question']}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=500, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)
    gen_tokens = out.sequences[0, input_ids.shape[1]:]

    detector = PhaseSpaceRecurrenceDetector(tokenizer)
    for idx, tok in enumerate(gen_tokens):
        hs = out.hidden_states[idx][-1][0, -1, :]
        res = detector.step(tok.item(), hs)
        if res.is_loop:
            step_k = len(detector.steps) - 1
            print(f"[{pid}] Step {step_k} (tok {idx}): {res.reason}")


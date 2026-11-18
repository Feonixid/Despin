import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from despin.detector_v2 import PhaseSpaceRecurrenceDetector

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")

q = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every remaining egg at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

# Let's run greedy generation and inspect all steps and their pairwise similarities
with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=500, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)

gen_tokens = out.sequences[0, input_ids.shape[1]:]
text = tokenizer.decode(gen_tokens)

detector = PhaseSpaceRecurrenceDetector(tokenizer, recurrence_threshold=0.90, velocity_collapse_threshold=0.18)

for idx, tok in enumerate(gen_tokens):
    hs = out.hidden_states[idx][-1][0, -1, :]
    res = detector.step(tok.item(), hs)
    if res.is_loop:
        step_k = len(detector.steps) - 1
        print(f"Token {idx:3d} (Step {step_k:2d}): LOOP DETECTED! {res.reason}")
        # print text of step k and historical steps
        k_text = detector.steps[-1].text.strip().replace('\n', ' ')
        print(f"   Step {step_k}: '{k_text[:60]}'")
        if res.pivot_token_idx >= 0:
            p_text = tokenizer.decode(gen_tokens[res.pivot_token_idx:res.pivot_token_idx+30]).replace('\n', ' ')
            print(f"   Pivot text at tok {res.pivot_token_idx}: '{p_text}'")


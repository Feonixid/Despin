import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from despin.detector_v2 import PhaseSpaceRecurrenceDetector

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")

q = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every remaining egg at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=450, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)

gen_tokens = out.sequences[0, input_ids.shape[1]:]

detector = PhaseSpaceRecurrenceDetector(tokenizer, recurrence_threshold=0.90, velocity_collapse_threshold=0.18)

for idx, tok in enumerate(gen_tokens):
    hs = out.hidden_states[idx][-1][0, -1, :]
    detector.step(tok.item(), hs)

print(f"Total steps captured: {len(detector.steps)}")
for s in detector.steps:
    print(f"Step {s.step_id:2d} (tok {s.start_token_idx:3d}-{s.end_token_idx:3d}): vel={s.velocity:.4f} | text='{s.text.strip().replace(chr(10), ' ')[:50]}'")


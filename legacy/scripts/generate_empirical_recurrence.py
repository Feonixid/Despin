import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 300,
})

def segment_into_steps(text, min_chars=30):
    parts = re.split(r'(?<=[.!?\n])\s+', text)
    steps = []
    buf = ""
    for p in parts:
        buf += (" " if buf else "") + p
        if len(buf) >= min_chars:
            steps.append(buf.strip())
            buf = ""
    if buf.strip():
        steps.append(buf.strip())
    return steps

def generate_fig3_empirical():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading {model_id} for empirical phase space plot...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")

    cases = [
        {
            "name": "Limit Cycle Pathology (Looping Reasoning)",
            "q": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every remaining egg at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "max_tokens": 400,
        },
        {
            "name": "Healthy Trajectory (Progressive Reasoning)",
            "q": "Solve for x: 2x + 5 = 15. What is x?",
            "max_tokens": 200,
        }
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    for idx, case in enumerate(cases):
        messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {case['q']}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=case["max_tokens"],
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        gen_ids = out.sequences[0, prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        all_hidden = []
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            for step_hs in out.hidden_states:
                last_layer = step_hs[-1]
                last_token_hs = last_layer[0, -1, :]
                all_hidden.append(last_token_hs.cpu().float())

        steps = segment_into_steps(gen_text, min_chars=35)
        step_token_indices = []
        token_cursor = 0
        for step_text in steps:
            step_toks = tokenizer.encode(step_text, add_special_tokens=False)
            end_idx = min(token_cursor + len(step_toks), len(all_hidden) - 1)
            step_token_indices.append(end_idx)
            token_cursor = end_idx + 1

        step_reprs = [all_hidden[i] for i in step_token_indices if i < len(all_hidden)]
        n = min(len(step_reprs), 16)
        step_reprs = step_reprs[:n]

        cos_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cos_mat[i, j] = F.cosine_similarity(
                    step_reprs[i].unsqueeze(0), step_reprs[j].unsqueeze(0), dim=-1
                ).item()

        im = axes[idx].imshow(cos_mat, cmap="magma", vmin=0.4, vmax=1.0)
        axes[idx].set_title(f"{case['name']}\n({n} Reasoning Steps)", fontsize=11, weight="bold")
        axes[idx].set_xlabel("Reasoning Step Index $j$")
        axes[idx].set_ylabel("Reasoning Step Index $k$")
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04, label="Cosine Similarity")

    plt.tight_layout()
    out_path = 'figures_v2/fig3_empirical_recurrence.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

    artifact_dir = "C:/Users/User/.gemini/antigravity/brain/275d4ee5-d34a-4a1f-acdc-dcbf60c53518/figures_v2"
    import shutil
    shutil.copy(out_path, os.path.join(artifact_dir, "fig3_empirical_recurrence.png"))
    print(f"Copied to {artifact_dir}")

if __name__ == '__main__':
    generate_fig3_empirical()


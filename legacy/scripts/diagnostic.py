"""
Diagnostic: Probe hidden state dynamics during reasoning loops.
Captures step-level representations, cosine similarity matrices,
and velocity profiles to inform the principled detector design.
"""
import torch
import torch.nn.functional as F
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

def segment_into_steps(text, min_chars=30):
    """Split reasoning trace into logical steps at sentence boundaries."""
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

def run_diagnostic():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")

    # Use a problem known to trigger loops from v1 results
    problems = [
        {
            "label": "GSM8K_eggs (looped in v1)",
            "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every remaining egg at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "answer": "18",
        },
        {
            "label": "MATH_NumberTheory (looped in v1)",
            "question": "How many positive whole numbers have cube roots that are less than 10?",
            "answer": "999",
        },
        {
            "label": "Easy_algebra (clean in v1)",
            "question": "Solve for x: 2x + 5 = 15. What is x?",
            "answer": "5",
        },
    ]

    for prob in problems:
        print(f"\n{'='*80}")
        print(f"Problem: {prob['label']}")
        print(f"{'='*80}")

        messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {prob['question']}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        prompt_len = input_ids.shape[1]

        # Generate with hidden state logging
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=2048,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        gen_ids = out.sequences[0, prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        print(f"Generated {len(gen_ids)} tokens")

        # Extract per-token hidden states (last layer)
        # out.hidden_states is a tuple of (step, layer, [batch, seq_len, hidden_dim])
        # For generate, each step has the hidden states of that decoding step
        all_hidden = []
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            for step_hs in out.hidden_states:
                # step_hs is a tuple of layer hidden states
                # Take last layer, last token
                last_layer = step_hs[-1]  # [batch, seq_in_step, hidden_dim]
                last_token_hs = last_layer[0, -1, :]  # [hidden_dim]
                all_hidden.append(last_token_hs.cpu().float())

        print(f"Captured {len(all_hidden)} hidden state vectors (dim={all_hidden[0].shape[0] if all_hidden else '?'})")

        if len(all_hidden) < 10:
            print("Too few hidden states for analysis, skipping.")
            continue

        # Segment text into steps and map to token ranges
        steps = segment_into_steps(gen_text, min_chars=40)
        print(f"Segmented into {len(steps)} reasoning steps")

        # Map steps to approximate token indices
        step_token_indices = []
        token_cursor = 0
        for step_text in steps:
            step_toks = tokenizer.encode(step_text, add_special_tokens=False)
            end_idx = min(token_cursor + len(step_toks), len(all_hidden) - 1)
            step_token_indices.append(end_idx)
            token_cursor = end_idx + 1

        # Get step-level hidden states (at step boundaries)
        step_reprs = []
        for idx in step_token_indices:
            if idx < len(all_hidden):
                step_reprs.append(all_hidden[idx])

        n_steps = len(step_reprs)
        if n_steps < 4:
            print("Too few steps for dynamics analysis.")
            continue

        # === ANALYSIS 1: Pairwise Cosine Similarity Matrix ===
        print(f"\n--- Pairwise Cosine Similarity (Step-Level, {n_steps} steps) ---")
        cos_matrix = torch.zeros(n_steps, n_steps)
        for i in range(n_steps):
            for j in range(n_steps):
                cos_matrix[i, j] = F.cosine_similarity(
                    step_reprs[i].unsqueeze(0), step_reprs[j].unsqueeze(0), dim=-1
                ).item()

        # Print abbreviated similarity matrix
        print(f"{'':>5s}", end="")
        for j in range(min(n_steps, 20)):
            print(f"  S{j:<3d}", end="")
        print()
        for i in range(min(n_steps, 20)):
            print(f"S{i:<3d} ", end="")
            for j in range(min(n_steps, 20)):
                val = cos_matrix[i, j].item()
                marker = " " if i == j else ("*" if val > 0.92 else ("+" if val > 0.85 else " "))
                print(f" {val:.3f}{marker}", end="")
            print()

        # === ANALYSIS 2: Angular Velocity Profile ===
        print(f"\n--- Angular Velocity Profile (1 - cos_sim between consecutive steps) ---")
        velocities = []
        for k in range(1, n_steps):
            v = 1.0 - F.cosine_similarity(
                step_reprs[k].unsqueeze(0), step_reprs[k-1].unsqueeze(0), dim=-1
            ).item()
            velocities.append(v)

        for k, v in enumerate(velocities):
            bar = "#" * int(v * 200)
            stag = " [STAGNANT]" if v < 0.02 else ""
            print(f"  Step {k} -> {k+1}: velocity={v:.4f} {bar}{stag}")

        mean_v = np.mean(velocities)
        std_v = np.std(velocities)
        min_v = np.min(velocities)
        print(f"  Mean velocity: {mean_v:.4f}, Std: {std_v:.4f}, Min: {min_v:.4f}")

        # === ANALYSIS 3: Recurrence Detection ===
        print(f"\n--- Recurrence Analysis ---")
        # For each step, find the most similar earlier step (excluding self and neighbors)
        for k in range(3, n_steps):
            best_sim = -1
            best_j = -1
            for j in range(k - 2):  # skip immediate neighbors
                sim = cos_matrix[k, j].item()
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            if best_sim > 0.88:
                period = k - best_j
                # Check if periodic: does step k-period also match step k-2*period?
                periodic = False
                if k - period >= 0 and best_j - period >= 0:
                    prev_sim = cos_matrix[k - period, best_j - period].item()
                    periodic = prev_sim > 0.88
                tag = "PERIODIC ORBIT" if periodic else "RECURRENCE"
                print(f"  Step {k} ~ Step {best_j} (sim={best_sim:.4f}, period={period}) [{tag}]")

        # === ANALYSIS 4: Print step text snippets ===
        print(f"\n--- Step Text Snippets ---")
        for k, step in enumerate(steps[:20]):
            snippet = step[:80].replace("\n", " ")
            print(f"  Step {k}: {snippet}...")

        print()

if __name__ == "__main__":
    run_diagnostic()


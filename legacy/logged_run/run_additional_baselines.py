import os
import json
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from despin.engine_v2 import DespinEngineV2
from despin.evaluator import MathEvaluator

class BaselineRunner:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eos_token_ids = set([tokenizer.eos_token_id])
        if hasattr(model.config, "eos_token_id") and model.config.eos_token_id is not None:
            if isinstance(model.config.eos_token_id, list):
                self.eos_token_ids.update(model.config.eos_token_id)
            else:
                self.eos_token_ids.add(model.config.eos_token_id)

    @torch.no_grad()
    def generate_repetition_penalty(self, prompt: str, penalty: float = 1.15, max_new_tokens: int = 2048):
        start_time = time.perf_counter()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        outputs = self.model(input_ids=input_ids, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        generated_tokens = []
        token_set = set()

        while len(generated_tokens) < max_new_tokens:
            logits = next_token_logits.clone()

            if token_set:
                unique_tokens = torch.tensor([list(token_set)], device=self.device)
                score = torch.gather(logits, 1, unique_tokens)
                score = torch.where(score > 0, score / penalty, score * penalty)
                logits.scatter_(1, unique_tokens, score)

            next_token_id = int(torch.argmax(logits, dim=-1).item())
            generated_tokens.append(next_token_id)
            token_set.add(next_token_id)

            if next_token_id in self.eos_token_ids:
                break

            current_token_tensor = torch.tensor([[next_token_id]], device=self.device)
            outputs = self.model(
                input_ids=current_token_tensor,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        total_latency_s = time.perf_counter() - start_time
        final_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return final_text, len(generated_tokens), total_latency_s


def run_baselines():
    with open("results_v2/massive_sweep_problems.json", "r", encoding="utf-8") as f:
        problems = json.load(f)

    # Load existing greedy and despin results
    with open("results_v2/massive_sweep_final.json", "r", encoding="utf-8") as f:
        existing_data = json.load(f)

    existing_by_id = {r["problem"]["id"]: r for r in existing_data["records"]}

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading {model_id} on GPU...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    engine = DespinEngineV2(
        model=model,
        tokenizer=tokenizer,
        recurrence_threshold=0.90,
        velocity_collapse_threshold=0.18,
    )
    rep_runner = BaselineRunner(model=model, tokenizer=tokenizer)

    out_file = "results_v2/massive_sweep_all_baselines.json"
    records = []

    # Running totals
    stats = {
        "Greedy (T=0.0)": {"correct": 0, "tokens": 0, "lat": 0.0},
        "Sampling (T=0.6, p=0.95)": {"correct": 0, "tokens": 0, "lat": 0.0},
        "Repetition Penalty (1.15)": {"correct": 0, "tokens": 0, "lat": 0.0},
        "Passive Early-Exit (Wei et al. 2025)": {"correct": 0, "tokens": 0, "lat": 0.0},
        "Tri-Regime DESPIN (Ours)": {"correct": 0, "tokens": 0, "lat": 0.0},
    }

    print(f"\n{'='*80}")
    print("RUNNING ADDITIONAL BASELINES ON 60-PROBLEM BENCHMARK")
    print("Baselines: Sampling (T=0.6), Repetition Penalty (1.15), Passive Early-Exit")
    print(f"{'='*80}\n", flush=True)

    for i, prob in enumerate(problems):
        pid = prob["id"]
        bench = prob["benchmark"]
        q = prob["question"]
        gt = prob["ground_truth"]

        messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 1. Existing Greedy & DESPIN
        rec_exist = existing_by_id[pid]
        g_corr = rec_exist["greedy"]["correct"]
        g_tok = rec_exist["greedy"]["tokens"]
        g_lat = rec_exist["greedy"]["latency_s"]

        d_corr = rec_exist["despin"]["correct"]
        d_tok = rec_exist["despin"]["tokens"]
        d_lat = rec_exist["despin"]["latency_s"]

        # 2. Nucleus Sampling (T=0.6, p=0.95)
        t0 = time.time()
        res_samp = engine.generate(
            prompt=prompt,
            max_new_tokens=2048,
            base_temperature=0.6,
            top_p=0.95,
            enable_despin=False,
        )
        lat_samp = time.time() - t0
        pred_samp = MathEvaluator.extract_answer(res_samp.text)
        corr_samp = MathEvaluator.is_equiv(pred_samp, gt)

        # 3. Repetition Penalty (1.15, T=0.0)
        t1 = time.time()
        text_rep, tok_rep, lat_rep = rep_runner.generate_repetition_penalty(prompt=prompt, penalty=1.15)
        pred_rep = MathEvaluator.extract_answer(text_rep)
        corr_rep = MathEvaluator.is_equiv(pred_rep, gt)

        # 4. Passive Early-Exit (Wei et al. 2025)
        t2 = time.time()
        res_exit = engine.generate(
            prompt=prompt,
            max_new_tokens=2048,
            base_temperature=0.0,
            enable_despin=False,
            passive_early_exit=True,
        )
        lat_exit = time.time() - t2
        pred_exit = MathEvaluator.extract_answer(res_exit.text)
        corr_exit = MathEvaluator.is_equiv(pred_exit, gt)

        # Update stats
        stats["Greedy (T=0.0)"]["correct"] += int(g_corr)
        stats["Greedy (T=0.0)"]["tokens"] += g_tok
        stats["Greedy (T=0.0)"]["lat"] += g_lat

        stats["Sampling (T=0.6, p=0.95)"]["correct"] += int(corr_samp)
        stats["Sampling (T=0.6, p=0.95)"]["tokens"] += res_samp.final_tokens_count
        stats["Sampling (T=0.6, p=0.95)"]["lat"] += lat_samp

        stats["Repetition Penalty (1.15)"]["correct"] += int(corr_rep)
        stats["Repetition Penalty (1.15)"]["tokens"] += tok_rep
        stats["Repetition Penalty (1.15)"]["lat"] += lat_rep

        stats["Passive Early-Exit (Wei et al. 2025)"]["correct"] += int(corr_exit)
        stats["Passive Early-Exit (Wei et al. 2025)"]["tokens"] += res_exit.final_tokens_count
        stats["Passive Early-Exit (Wei et al. 2025)"]["lat"] += lat_exit

        stats["Tri-Regime DESPIN (Ours)"]["correct"] += int(d_corr)
        stats["Tri-Regime DESPIN (Ours)"]["tokens"] += d_tok
        stats["Tri-Regime DESPIN (Ours)"]["lat"] += d_lat

        print(
            f"[{i+1:02d}/60] {pid:10s} | "
            f"Greedy: {'P' if g_corr else 'F'} ({g_tok}t) | "
            f"Samp: {'P' if corr_samp else 'F'} ({res_samp.final_tokens_count}t) | "
            f"RepPen: {'P' if corr_rep else 'F'} ({tok_rep}t) | "
            f"EarlyExit: {'P' if corr_exit else 'F'} ({res_exit.final_tokens_count}t) | "
            f"DESPIN: {'P' if d_corr else 'F'} ({d_tok}t)",
            flush=True,
        )

        records.append({
            "problem": prob,
            "greedy": {"correct": g_corr, "tokens": g_tok, "latency_s": g_lat},
            "sampling": {"correct": corr_samp, "tokens": res_samp.final_tokens_count, "latency_s": lat_samp},
            "repetition_penalty": {"correct": corr_rep, "tokens": tok_rep, "latency_s": lat_rep},
            "passive_early_exit": {"correct": corr_exit, "tokens": res_exit.final_tokens_count, "latency_s": lat_exit},
            "despin": rec_exist["despin"],
        })

        # Save live checkpoint
        n_done = len(records)
        summary = []
        for m, s in stats.items():
            acc = (s["correct"] / n_done) * 100
            avg_tok = s["tokens"] / n_done
            avg_lat = s["lat"] / n_done
            tes = acc / (avg_tok / 1000.0) if avg_tok > 0 else 0.0
            summary.append({
                "Method": m,
                "Pass@1 Acc (%)": round(acc, 2),
                "Avg Tokens": round(avg_tok, 1),
                "Avg Latency (s)": round(avg_lat, 2),
                "Token Efficiency Score (TES)": round(tes, 2),
            })

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "completed": n_done,
                "total": len(problems),
                "summary": summary,
                "records": records,
            }, f, indent=2)

    print("\n" + "="*80)
    print("ALL BASELINES COMPLETED ACROSS 60 PROBLEMS")
    print("="*80)
    for row in summary:
        print(f"{row['Method']:38s} | Acc: {row['Pass@1 Acc (%)']:5.2f}% | Tok: {row['Avg Tokens']:6.1f} | TES: {row['Token Efficiency Score (TES)']:5.2f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_baselines()


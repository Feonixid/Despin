import os
import json
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from despin.engine_v2 import GenerationResultV2
from despin.detector_v2 import PhaseSpaceRecurrenceDetector, PhaseSpaceDetectionResult
from despin.evaluator import MathEvaluator
from despin.kv_cache import KVCacheManager

class GramSchmidtDespinEngine:
    """
    DESPIN with Gram-Schmidt Orthogonal Projective Deflation:
    h_t^perp = h_t - beta * max(0, <h_t, u>) * u
    => delta_z = beta * max(0, <h_t, u>) * ReLU(W_u * u)
    Coupled with Dual-Regime Phase-Space Gated Early Closure (PGEC).
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        recurrence_threshold: float = 0.90,
        velocity_collapse_threshold: float = 0.18,
        max_rollbacks: int = 2,
        orthogonal_stiffness: float = 12.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.recurrence_threshold = recurrence_threshold
        self.velocity_collapse_threshold = velocity_collapse_threshold
        self.max_rollbacks = max_rollbacks
        self.orthogonal_stiffness = orthogonal_stiffness
        self.W_u = model.lm_head.weight

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        enable_pgec: bool = True,
    ) -> GenerationResultV2:
        start_time = time.perf_counter()
        detector = PhaseSpaceRecurrenceDetector(
            tokenizer=self.tokenizer,
            recurrence_threshold=self.recurrence_threshold,
            velocity_collapse_threshold=self.velocity_collapse_threshold,
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]

        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]

        generated_tokens = []
        total_tokens_generated = 0
        pruned_tokens_count = 0
        rollback_history = []
        total_rollback_ms = 0.0

        # Active Gram-Schmidt attractor unit vector
        active_u_attractor = None
        active_proj_Wu = None

        eos_token_ids = set([self.tokenizer.eos_token_id])
        if hasattr(self.model.config, "eos_token_id") and self.model.config.eos_token_id is not None:
            if isinstance(self.model.config.eos_token_id, list):
                eos_token_ids.update(self.model.config.eos_token_id)
            else:
                eos_token_ids.add(self.model.config.eos_token_id)

        candidate_answer_found = False

        while len(generated_tokens) < max_new_tokens:
            logits = next_token_logits.clone()

            # Dynamic Gram-Schmidt Orthogonal Deflation
            if active_u_attractor is not None and last_hidden_state is not None:
                h_norm = last_hidden_state.squeeze()
                # Compute instantaneous manifold alignment: <h_t, u_attractor>
                alignment = torch.dot(h_norm, active_u_attractor).item()
                if alignment > 0:
                    penalty = (self.orthogonal_stiffness * alignment) * active_proj_Wu
                    logits -= penalty
                else:
                    # Trajectory has naturally escaped the attractor cone!
                    active_u_attractor = None
                    active_proj_Wu = None

            next_token_id = int(torch.argmax(logits, dim=-1).item())
            generated_tokens.append(next_token_id)
            total_tokens_generated += 1

            if next_token_id in eos_token_ids:
                break

            current_token_tensor = torch.tensor([[next_token_id]], device=self.device)
            outputs = self.model(
                input_ids=current_token_tensor,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            last_hidden_state = outputs.hidden_states[-1][:, -1, :] if outputs.hidden_states else None

            # Online Phase-Space Detector step
            loop_res: PhaseSpaceDetectionResult = detector.step(
                token_id=next_token_id,
                hidden_state=last_hidden_state,
            )

            if detector.candidate_answer_step_idx is not None:
                candidate_answer_found = True

            if loop_res.is_loop:
                # REGIME 2: Post-Solution Overthinking -> Phase-Space Gated Early Closure (PGEC)
                if enable_pgec and candidate_answer_found:
                    rollback_history.append({
                        "type": "pgec_early_closure",
                        "at_token": len(generated_tokens),
                        "reason": loop_res.reason,
                    })
                    break

                # REGIME 1: Pre-Solution Impasse -> Gram-Schmidt Attractor Deflation KV Rollback
                if len(rollback_history) < self.max_rollbacks:
                    pivot_idx = max(0, loop_res.pivot_token_idx)
                    tokens_to_prune = len(generated_tokens) - pivot_idx

                    if tokens_to_prune > 0 and pivot_idx < len(generated_tokens):
                        target_kv_len = prompt_len + pivot_idx

                        past_key_values, slice_ms = KVCacheManager.rollback(
                            past_key_values, target_kv_len
                        )
                        total_rollback_ms += slice_ms

                        if loop_res.attractor_vector is not None:
                            u_attractor = loop_res.attractor_vector.to(self.device).to(self.W_u.dtype)
                            norm_u = torch.norm(u_attractor)
                            if norm_u > 1e-6:
                                active_u_attractor = u_attractor / norm_u
                                # Precompute W_u * u_attractor for fast projection
                                proj = F.linear(active_u_attractor.unsqueeze(0), self.W_u).squeeze(0)
                                active_proj_Wu = F.relu(proj)

                        generated_tokens = generated_tokens[:pivot_idx]
                        pruned_tokens_count += tokens_to_prune

                        # Re-evaluate pivot token
                        last_valid_token_id = (
                            generated_tokens[-1]
                            if len(generated_tokens) > 0
                            else input_ids[0, -1].item()
                        )
                        pivot_in = torch.tensor([[last_valid_token_id]], device=self.device)
                        target_prev_len = target_kv_len - 1
                        past_key_values, _ = KVCacheManager.rollback(past_key_values, target_prev_len)

                        outputs = self.model(
                            input_ids=pivot_in,
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        past_key_values = outputs.past_key_values
                        next_token_logits = outputs.logits[:, -1, :]
                        last_hidden_state = outputs.hidden_states[-1][:, -1, :] if outputs.hidden_states else None

                        detector.reset()
                        for t in generated_tokens:
                            detector.step(t)
                        candidate_answer_found = detector.candidate_answer_step_idx is not None

                        rollback_history.append({
                            "type": "gram_schmidt_asd_rollback",
                            "at_token": len(generated_tokens) + tokens_to_prune,
                            "rewound_to": pivot_idx,
                            "tokens_pruned": tokens_to_prune,
                            "reason": loop_res.reason,
                            "slice_overhead_ms": slice_ms,
                        })

        total_latency_s = time.perf_counter() - start_time
        final_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return GenerationResultV2(
            text=final_text,
            tokens=generated_tokens,
            total_tokens_generated=total_tokens_generated,
            final_tokens_count=len(generated_tokens),
            pruned_tokens_count=pruned_tokens_count,
            rollbacks_count=len(rollback_history),
            rollback_history=rollback_history,
            generation_latency_s=total_latency_s,
            total_rollback_overhead_ms=total_rollback_ms,
            model_name=getattr(self.model.config, "_name_or_path", "unknown"),
        )

def run_test():
    with open('results_v2/benchmark_v2_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    problems = [r['problem'] for r in data['records']]

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    engine = GramSchmidtDespinEngine(
        model=model,
        tokenizer=tokenizer,
        recurrence_threshold=0.90,
        velocity_collapse_threshold=0.18,
        max_rollbacks=2,
        orthogonal_stiffness=12.0,
    )

    correct_count = 0
    total_tokens = 0
    total_latency = 0.0
    results = []

    print("=== Running DESPIN-v4 with Gram-Schmidt Orthogonal Projective Deflation ===", flush=True)

    for idx, prob in enumerate(problems):
        q = prob['question']
        gt = prob['ground_truth']
        messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        t0 = time.time()
        res = engine.generate(prompt=prompt, max_new_tokens=2048, enable_pgec=True)
        elapsed = time.time() - t0

        pred = MathEvaluator.extract_answer(res.text)
        correct = MathEvaluator.is_equiv(pred, gt)

        if correct:
            correct_count += 1
        total_tokens += res.total_tokens_generated
        total_latency += elapsed

        actions = [h['type'] for h in res.rollback_history]
        results.append({
            "problem_id": prob['id'],
            "correct": correct,
            "pred": pred,
            "tokens": res.total_tokens_generated,
            "pruned": res.pruned_tokens_count,
            "actions": actions,
            "latency_s": elapsed,
        })

        status = "PASS" if correct else "FAIL"
        print(f"[{idx+1:2d}/{len(problems)}] {prob['id']:12s}: {status} | Pred: {str(pred)[:15]:15s} | GT: {str(gt)[:10]:10s} | Tok: {res.total_tokens_generated:4d} | Actions: {actions}", flush=True)

    acc = correct_count / len(problems) * 100.0
    avg_tokens = total_tokens / len(problems)
    avg_lat = total_latency / len(problems)
    tes = acc / (avg_tokens / 1000.0)

    print("\n" + "=" * 75, flush=True)
    print(f"DESPIN-v4 (Gram-Schmidt Orthogonal Deflation + PGEC):", flush=True)
    print(f"Accuracy:   {acc:.2f}%", flush=True)
    print(f"Avg Tokens: {avg_tokens:.1f}", flush=True)
    print(f"Avg Lat:    {avg_lat:.2f}s", flush=True)
    print(f"TES:        {tes:.2f}", flush=True)
    print("=" * 75, flush=True)

    with open('results_v2/despin_gram_schmidt.json', 'w', encoding='utf-8') as f:
        json.dump({
            "accuracy": acc,
            "avg_tokens": avg_tokens,
            "avg_latency": avg_lat,
            "tes": tes,
            "records": results,
        }, f, indent=2)

if __name__ == "__main__":
    run_test()


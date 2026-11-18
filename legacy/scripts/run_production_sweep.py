import os
import json
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from despin.engine_v2 import GenerationResultV2
from despin.detector_v2 import PhaseSpaceStep, PhaseSpaceDetectionResult
from despin.evaluator import MathEvaluator
from despin.kv_cache import KVCacheManager

class ProductionPhaseSpaceDetector:
    """
    Production Phase-Space Recurrence Detector:
    - Warmup protection: first 80 tokens protected from premature intervention.
    - Calibrated limit-cycle detection: cos_sim >= 0.90 across non-adjacent steps.
    - Angular velocity collapse: v < 0.18 across consecutive steps.
    - Candidate answer tracking for Phase-Space Gated Early Closure (PGEC).
    """

    def __init__(
        self,
        tokenizer,
        recurrence_threshold: float = 0.90,
        velocity_collapse_threshold: float = 0.18,
        consecutive_slow_steps: int = 2,
        min_step_tokens: int = 6,
        warmup_tokens: int = 80,
    ):
        self.tokenizer = tokenizer
        self.recurrence_threshold = recurrence_threshold
        self.velocity_collapse_threshold = velocity_collapse_threshold
        self.consecutive_slow_steps = consecutive_slow_steps
        self.min_step_tokens = min_step_tokens
        self.warmup_tokens = warmup_tokens
        self.reset()

    def reset(self):
        self.steps = []
        self.current_step_tokens = []
        self.current_step_start_idx = 0
        self.total_tokens_seen = 0
        self.slow_step_streak = 0
        self.detected_loops_count = 0
        self.candidate_answer_step_idx = None

    def is_step_boundary(self, token_id: int, current_text: str) -> bool:
        token_str = self.tokenizer.decode([token_id])
        if "\n" in token_str or current_text.endswith((".", "!", "?", "...", ":\n")):
            return len(self.current_step_tokens) >= self.min_step_tokens
        return False

    def step(
        self,
        token_id: int,
        hidden_state: torch.Tensor = None,
    ) -> PhaseSpaceDetectionResult:
        self.current_step_tokens.append(token_id)
        current_token_idx = self.total_tokens_seen
        self.total_tokens_seen += 1

        step_text = self.tokenizer.decode(self.current_step_tokens, skip_special_tokens=True)
        if not self.is_step_boundary(token_id, step_text):
            return PhaseSpaceDetectionResult(
                is_loop=False, pivot_token_idx=-1, recurrence_sim=0.0
            )

        h_vec = hidden_state.detach().cpu().float().squeeze() if hidden_state is not None else torch.zeros(1536)

        velocity = 1.0
        if self.steps:
            prev_h = self.steps[-1].hidden_state
            cos_prev = F.cosine_similarity(h_vec.unsqueeze(0), prev_h.unsqueeze(0), dim=-1).item()
            velocity = 1.0 - cos_prev

        new_step = PhaseSpaceStep(
            step_id=len(self.steps),
            start_token_idx=self.current_step_start_idx,
            end_token_idx=current_token_idx,
            text=step_text,
            tokens=list(self.current_step_tokens),
            hidden_state=h_vec,
            velocity=velocity,
        )
        self.steps.append(new_step)
        self.current_step_tokens = []
        self.current_step_start_idx = current_token_idx + 1

        if "\\boxed{" in step_text or "the answer is" in step_text.lower():
            self.candidate_answer_step_idx = len(self.steps) - 1

        # Warmup protection
        if current_token_idx < self.warmup_tokens:
            return PhaseSpaceDetectionResult(is_loop=False, pivot_token_idx=-1, recurrence_sim=0.0)

        best_sim = -1.0
        best_j = -1
        k = len(self.steps) - 1
        for j in range(k - 2):
            sim = F.cosine_similarity(h_vec.unsqueeze(0), self.steps[j].hidden_state.unsqueeze(0), dim=-1).item()
            if sim > best_sim:
                best_sim = sim
                best_j = j

        # Limit Cycle Recurrence Condition
        if best_sim >= self.recurrence_threshold:
            self.detected_loops_count += 1
            pivot_step = self.steps[best_j]
            pivot_token_idx = pivot_step.start_token_idx
            diff = h_vec - pivot_step.hidden_state
            norm = torch.norm(diff, p=2)
            u_attractor = (diff / norm) if norm > 1e-6 else torch.zeros_like(diff)
            return PhaseSpaceDetectionResult(
                is_loop=True,
                pivot_token_idx=pivot_token_idx,
                recurrence_sim=best_sim,
                loop_type="limit_cycle",
                reason=f"Recurrence: Step {k} ~ Step {best_j} (sim={best_sim:.3f})",
                attractor_vector=u_attractor,
            )

        # Velocity Collapse Condition
        if velocity < self.velocity_collapse_threshold:
            self.slow_step_streak += 1
        else:
            self.slow_step_streak = max(0, self.slow_step_streak - 1)

        if self.slow_step_streak >= self.consecutive_slow_steps:
            self.detected_loops_count += 1
            self.slow_step_streak = 0
            freeze_start_step = self.steps[-self.consecutive_slow_steps]
            pivot_token_idx = freeze_start_step.start_token_idx
            diff = h_vec - freeze_start_step.hidden_state
            norm = torch.norm(diff, p=2)
            u_attractor = (diff / norm) if norm > 1e-6 else torch.zeros_like(diff)
            return PhaseSpaceDetectionResult(
                is_loop=True,
                pivot_token_idx=pivot_token_idx,
                recurrence_sim=1.0 - velocity,
                loop_type="velocity_collapse",
                reason=f"Velocity collapse: v={velocity:.4f}",
                attractor_vector=u_attractor,
            )

        return PhaseSpaceDetectionResult(is_loop=False, pivot_token_idx=-1, recurrence_sim=0.0)


class ProductionEngine:
    def __init__(
        self,
        model,
        tokenizer,
        device="cuda",
        recurrence_threshold=0.90,
        velocity_collapse_threshold=0.18,
        consecutive_slow_steps=2,
        warmup_tokens=80,
        max_rollbacks=2,
        deflation_gamma=15.0,
        exploration_steps=6,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.recurrence_threshold = recurrence_threshold
        self.velocity_collapse_threshold = velocity_collapse_threshold
        self.consecutive_slow_steps = consecutive_slow_steps
        self.warmup_tokens = warmup_tokens
        self.max_rollbacks = max_rollbacks
        self.deflation_gamma = deflation_gamma
        self.exploration_steps = exploration_steps
        self.W_u = model.lm_head.weight

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=2048, enable_pgec=True):
        start_time = time.perf_counter()
        detector = ProductionPhaseSpaceDetector(
            tokenizer=self.tokenizer,
            recurrence_threshold=self.recurrence_threshold,
            velocity_collapse_threshold=self.velocity_collapse_threshold,
            consecutive_slow_steps=self.consecutive_slow_steps,
            warmup_tokens=self.warmup_tokens,
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

        active_deflation_penalty = None
        deflation_decay_counter = 0
        candidate_answer_found = False

        eos_token_ids = set([self.tokenizer.eos_token_id])
        if hasattr(self.model.config, "eos_token_id") and self.model.config.eos_token_id is not None:
            if isinstance(self.model.config.eos_token_id, list):
                eos_token_ids.update(self.model.config.eos_token_id)
            else:
                eos_token_ids.add(self.model.config.eos_token_id)

        while len(generated_tokens) < max_new_tokens:
            logits = next_token_logits.clone()

            if active_deflation_penalty is not None and deflation_decay_counter > 0:
                decay_weight = deflation_decay_counter / float(self.exploration_steps)
                logits -= active_deflation_penalty * decay_weight
                deflation_decay_counter -= 1

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

            loop_res = detector.step(next_token_id, last_hidden_state)

            if detector.candidate_answer_step_idx is not None:
                candidate_answer_found = True

            if loop_res.is_loop:
                # REGIME 2: Post-Solution Overthinking -> Phase-Space Gated Early Closure
                if enable_pgec and candidate_answer_found:
                    rollback_history.append({
                        "type": "pgec_early_closure",
                        "at_token": len(generated_tokens),
                        "reason": loop_res.reason,
                    })
                    break

                # REGIME 1: Pre-Solution Impasse -> ASD Rollback
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
                                u_unit = u_attractor / norm_u
                                proj = F.linear(u_unit.unsqueeze(0), self.W_u).squeeze(0)
                                active_deflation_penalty = self.deflation_gamma * F.relu(proj)
                                deflation_decay_counter = self.exploration_steps

                        generated_tokens = generated_tokens[:pivot_idx]
                        pruned_tokens_count += tokens_to_prune

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
                            "type": "asd_rollback",
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

def run_production():
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

    engine = ProductionEngine(
        model=model,
        tokenizer=tokenizer,
        recurrence_threshold=0.90,
        velocity_collapse_threshold=0.18,
        consecutive_slow_steps=2,
        warmup_tokens=80,
        max_rollbacks=2,
        deflation_gamma=15.0,
        exploration_steps=6,
    )

    correct_count = 0
    total_tokens = 0
    total_latency = 0.0
    results = []

    print("=== Running Production DESPIN Benchmark ===", flush=True)

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
    print(f"Production DESPIN Results:", flush=True)
    print(f"Accuracy:   {acc:.2f}%", flush=True)
    print(f"Avg Tokens: {avg_tokens:.1f}", flush=True)
    print(f"Avg Lat:    {avg_lat:.2f}s", flush=True)
    print(f"TES:        {tes:.2f}", flush=True)
    print("=" * 75, flush=True)

    with open('results_v2/despin_production_final.json', 'w', encoding='utf-8') as f:
        json.dump({
            "accuracy": acc,
            "avg_tokens": avg_tokens,
            "avg_latency": avg_lat,
            "tes": tes,
            "records": results,
        }, f, indent=2)

if __name__ == "__main__":
    run_production()


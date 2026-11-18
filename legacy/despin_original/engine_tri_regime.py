import os
import json
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from despin.detector_v2 import PhaseSpaceStep, PhaseSpaceDetectionResult
from despin.evaluator import MathEvaluator
from despin.kv_cache import KVCacheManager

class PrincipledTriRegimeDetector:
    """
    Principled Phase-Space Trajectory Detector with Tri-Regime Control:
    1. Deliberation Phase: Inside <think>...</think>.
    2. Presentation Phase: After </think>.
    3. Candidate tracking: Specifically checks for \\boxed{...} formulation.
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
        self.has_boxed_answer = False
        self.in_think_phase = True

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

        if "</think>" in step_text:
            self.in_think_phase = False

        if "\\boxed{" in step_text:
            self.has_boxed_answer = True

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

        # Check full text so far for boxed answer
        if "\\boxed{" in step_text:
            self.has_boxed_answer = True

        # Warmup protection
        if current_token_idx < self.warmup_tokens:
            return PhaseSpaceDetectionResult(is_loop=False, pivot_token_idx=-1, recurrence_sim=0.0)

        # Condition A: Limit Cycle Recurrence Scan
        best_sim = -1.0
        best_j = -1
        k = len(self.steps) - 1
        for j in range(k - 2):
            sim = F.cosine_similarity(h_vec.unsqueeze(0), self.steps[j].hidden_state.unsqueeze(0), dim=-1).item()
            if sim > best_sim:
                best_sim = sim
                best_j = j

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

        # Condition B: Velocity Collapse
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


class TriRegimeDespinEngine:
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
    def generate(self, prompt: str, max_new_tokens=2048):
        start_time = time.perf_counter()
        detector = PrincipledTriRegimeDetector(
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

            if loop_res.is_loop:
                # REGIME 2: Post-Solution Overthinking (boxed answer already exists) -> PGEC
                if detector.has_boxed_answer:
                    rollback_history.append({
                        "type": "pgec_early_closure",
                        "at_token": len(generated_tokens),
                        "reason": f"Post-Solution Loop: {loop_res.reason} (Boxed=True)",
                    })
                    break

                # REGIME 3: Presentation Phase without boxed answer yet -> PASS THROUGH
                # (Model is formatting final steps, let it output \\boxed{} naturally)
                if not detector.in_think_phase:
                    continue

                # REGIME 1: Pre-Solution Impasse inside Deliberation Phase -> ASD Rollback
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

        from despin.engine_v2 import GenerationResultV2
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


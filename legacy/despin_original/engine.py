import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F

from .detector import AttractorDetector, LoopDetectionResult
from .kv_cache import KVCacheManager


@dataclass
class GenerationResult:
    text: str
    tokens: List[int]
    total_tokens_generated: int
    final_tokens_count: int
    pruned_tokens_count: int
    rollbacks_count: int
    rollback_history: List[Dict[str, Any]] = field(default_factory=list)
    generation_latency_s: float = 0.0
    total_rollback_overhead_ms: float = 0.0
    model_name: str = ""


class DespinEngine:
    """
    Inference Engine implementing DESPIN (Dynamic Epistemic Slicing and Pivot-Initiated Navigation).
    Conducts online attractor detection, O(1) KV-rollback, and directional anti-attractor branching.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ngram_repeat_threshold: float = 0.40,
        hesitation_burst_limit: int = 3,
        semantic_sim_threshold: float = 0.88,
        max_rollbacks: int = 4,
        logit_penalty_lambda: float = 8.0,
        exploration_temp_boost: float = 0.5,
        exploration_steps: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.ngram_repeat_threshold = ngram_repeat_threshold
        self.hesitation_burst_limit = hesitation_burst_limit
        self.semantic_sim_threshold = semantic_sim_threshold
        self.max_rollbacks = max_rollbacks
        self.logit_penalty_lambda = logit_penalty_lambda
        self.exploration_temp_boost = exploration_temp_boost
        self.exploration_steps = exploration_steps

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        base_temperature: float = 0.0,
        top_p: float = 1.0,
        enable_despin: bool = True,
        passive_early_exit: bool = False,
    ) -> GenerationResult:
        """
        Generate response with DESPIN active trajectory rescue.
        """
        start_time = time.perf_counter()
        detector = AttractorDetector(
            tokenizer=self.tokenizer,
            semantic_sim_threshold=self.semantic_sim_threshold,
            ngram_repeat_threshold=self.ngram_repeat_threshold,
            hesitation_burst_limit=self.hesitation_burst_limit,
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]

        # Prefill stage
        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        generated_tokens: List[int] = []
        total_tokens_generated = 0
        pruned_tokens_count = 0
        rollback_history: List[Dict[str, Any]] = []
        total_rollback_ms = 0.0

        current_token_tensor = None
        exploration_counter = 0
        banned_tokens_at_step: Dict[int, List[int]] = {}

        eos_token_ids = set([self.tokenizer.eos_token_id])
        if hasattr(self.model.config, "eos_token_id") and self.model.config.eos_token_id is not None:
            if isinstance(self.model.config.eos_token_id, list):
                eos_token_ids.update(self.model.config.eos_token_id)
            else:
                eos_token_ids.add(self.model.config.eos_token_id)

        # Autoregressive generation loop
        while len(generated_tokens) < max_new_tokens:
            curr_step_idx = len(generated_tokens)

            # Apply directional anti-attractor branching penalties if any
            logits = next_token_logits.clone()
            if curr_step_idx in banned_tokens_at_step:
                for b_tok in banned_tokens_at_step[curr_step_idx]:
                    logits[:, b_tok] -= self.logit_penalty_lambda

            # Compute sampling parameters
            curr_temp = base_temperature
            if exploration_counter > 0:
                curr_temp = max(curr_temp, self.exploration_temp_boost)
                exploration_counter -= 1

            if curr_temp == 0.0:
                next_token_id = int(torch.argmax(logits, dim=-1).item())
            else:
                scaled_logits = logits / curr_temp
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    scaled_logits[indices_to_remove] = -float("Inf")
                probs = F.softmax(scaled_logits, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())

            generated_tokens.append(next_token_id)
            total_tokens_generated += 1

            # Termination condition
            if next_token_id in eos_token_ids:
                break

            # Forward step with next token
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

            # Online Attractor Detection
            loop_res: LoopDetectionResult = detector.step(
                token_id=next_token_id,
                hidden_state=last_hidden_state,
            )

            # Check Passive Early-Exit baseline (Wei et al. 2025)
            if passive_early_exit and loop_res.is_loop:
                rollback_history.append({
                    "type": "passive_early_exit",
                    "step": len(generated_tokens),
                    "reason": loop_res.reason,
                })
                break

            # Check DESPIN Active Trajectory Intervention
            if enable_despin and loop_res.is_loop and len(rollback_history) < self.max_rollbacks:
                pivot_idx = max(0, loop_res.pivot_token_idx)
                tokens_to_prune = len(generated_tokens) - pivot_idx

                if tokens_to_prune > 0 and pivot_idx < len(generated_tokens):
                    target_kv_len = prompt_len + pivot_idx

                    # 1. Physical O(1) KV-Cache rollback
                    past_key_values, slice_ms = KVCacheManager.rollback(
                        past_key_values, target_kv_len
                    )
                    total_rollback_ms += slice_ms

                    # Mode A: Overthinking Re-verification after solution derivation
                    if loop_res.loop_type == "overthinking_reverification" and loop_res.candidate_answer:
                        generated_tokens = generated_tokens[:pivot_idx]
                        pruned_tokens_count += tokens_to_prune

                        # Inject clean conclusion
                        conclusion_str = f"\n</think>\nTherefore, the final answer is \\boxed{{{loop_res.candidate_answer}}}."
                        inj_tokens = self.tokenizer.encode(conclusion_str, add_special_tokens=False)
                        generated_tokens.extend(inj_tokens)
                        total_tokens_generated += len(inj_tokens)

                        rollback_history.append({
                            "type": "overthinking_despin_closure",
                            "at_token": len(generated_tokens) + tokens_to_prune,
                            "rewound_to": pivot_idx,
                            "tokens_pruned": tokens_to_prune,
                            "reason": loop_res.reason,
                            "slice_overhead_ms": slice_ms,
                            "candidate_answer": loop_res.candidate_answer,
                        })
                        break

                    # Mode B: Impasse Trajectory Rescue
                    else:
                        failed_first_token = (
                            loop_res.failed_tokens[0]
                            if loop_res.failed_tokens
                            else generated_tokens[pivot_idx]
                        )
                        if pivot_idx not in banned_tokens_at_step:
                            banned_tokens_at_step[pivot_idx] = []
                        banned_tokens_at_step[pivot_idx].append(failed_first_token)

                        generated_tokens = generated_tokens[:pivot_idx]
                        pruned_tokens_count += tokens_to_prune
                        exploration_counter = self.exploration_steps

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
                            return_dict=True,
                        )
                        past_key_values = outputs.past_key_values
                        next_token_logits = outputs.logits[:, -1, :]

                        detector.reset()
                        for t in generated_tokens:
                            detector.step(t)

                        rollback_history.append({
                            "type": "impasse_trajectory_rescue",
                            "at_token": len(generated_tokens) + tokens_to_prune,
                            "rewound_to": pivot_idx,
                            "tokens_pruned": tokens_to_prune,
                            "reason": loop_res.reason,
                            "slice_overhead_ms": slice_ms,
                        })

        total_latency_s = time.perf_counter() - start_time
        final_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return GenerationResult(
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

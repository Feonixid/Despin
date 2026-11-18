import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F

from .detector_v2 import PhaseSpaceRecurrenceDetector, PhaseSpaceDetectionResult
from .kv_cache import KVCacheManager


@dataclass
class GenerationResultV2:
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


class DespinEngineV2:
    """
    DESPIN Engine (Tri-Regime Deliberation-Gated Inference):
    Combines:
    1. Phase-Space Recurrence Detection (PSRD) with Warmup Protection.
    2. Tri-Regime Deliberation Manifold Control:
       - Regime 1: Pre-Solution Impasse (Inside <think>, no \\boxed{}) -> Attractor Subspace Deflation (ASD) + O(1) KV Slicing.
       - Regime 2: Post-Solution Overthinking (Has \\boxed{}) -> Phase-Space Gated Early Closure (PGEC).
       - Regime 3: Presentation Phase (Outside </think>, before \\boxed{}) -> Pass-through format generation.
    3. Continuous Attractor Subspace Deflation via Linear Unembedding Projection (W_u).
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        recurrence_threshold: float = 0.90,
        velocity_collapse_threshold: float = 0.18,
        consecutive_slow_steps: int = 2,
        warmup_tokens: int = 80,
        max_rollbacks: int = 2,
        deflation_gamma: float = 15.0,
        exploration_steps: int = 6,
        orthogonal_stiffness: Optional[float] = None,
        enable_pgec: bool = True,
        use_gram_schmidt: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.recurrence_threshold = recurrence_threshold
        self.velocity_collapse_threshold = velocity_collapse_threshold
        self.consecutive_slow_steps = consecutive_slow_steps
        self.warmup_tokens = warmup_tokens
        self.max_rollbacks = max_rollbacks
        self.deflation_gamma = orthogonal_stiffness if orthogonal_stiffness is not None else deflation_gamma
        self.exploration_steps = exploration_steps
        self.enable_pgec = enable_pgec
        self.use_gram_schmidt = use_gram_schmidt

        # Unembedding matrix for Attractor Subspace Deflation
        self.W_u = model.lm_head.weight  # [vocab_size, hidden_dim]

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        base_temperature: float = 0.0,
        top_p: float = 1.0,
        enable_despin: bool = True,
        passive_early_exit: bool = False,
        enable_pgec: Optional[bool] = None,
    ) -> GenerationResultV2:
        """
        Generate text autoregressively with Tri-Regime phase-space dynamic intervention.
        """
        if enable_pgec is None:
            enable_pgec = self.enable_pgec

        start_time = time.perf_counter()
        detector = PhaseSpaceRecurrenceDetector(
            tokenizer=self.tokenizer,
            recurrence_threshold=self.recurrence_threshold,
            velocity_collapse_threshold=self.velocity_collapse_threshold,
            consecutive_slow_steps=self.consecutive_slow_steps,
            warmup_tokens=self.warmup_tokens,
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
        last_hidden_state = outputs.hidden_states[-1][:, -1, :] if outputs.hidden_states else None

        generated_tokens: List[int] = []
        total_tokens_generated = 0
        pruned_tokens_count = 0
        rollback_history: List[Dict[str, Any]] = []
        total_rollback_ms = 0.0

        # Deflation control
        active_deflation_penalty: Optional[torch.Tensor] = None
        deflation_decay_counter = 0
        active_u_attractor: Optional[torch.Tensor] = None
        active_proj_Wu: Optional[torch.Tensor] = None

        eos_token_ids = set([self.tokenizer.eos_token_id])
        if hasattr(self.model.config, "eos_token_id") and self.model.config.eos_token_id is not None:
            if isinstance(self.model.config.eos_token_id, list):
                eos_token_ids.update(self.model.config.eos_token_id)
            else:
                eos_token_ids.add(self.model.config.eos_token_id)

        # Autoregressive generation loop
        while len(generated_tokens) < max_new_tokens:
            logits = next_token_logits.clone()

            # Apply attractor subspace deflation penalty
            if self.use_gram_schmidt and active_u_attractor is not None and last_hidden_state is not None:
                h_norm = last_hidden_state.squeeze()
                alignment = torch.dot(h_norm, active_u_attractor).item()
                if alignment > 0:
                    penalty = (self.deflation_gamma * alignment) * active_proj_Wu
                    logits -= penalty
                else:
                    active_u_attractor = None
                    active_proj_Wu = None
            elif active_deflation_penalty is not None and deflation_decay_counter > 0:
                decay_weight = deflation_decay_counter / float(self.exploration_steps)
                logits -= active_deflation_penalty * decay_weight
                deflation_decay_counter -= 1

            if base_temperature == 0.0:
                next_token_id = int(torch.argmax(logits, dim=-1).item())
            else:
                scaled_logits = logits / base_temperature
                probs = F.softmax(scaled_logits, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())

            generated_tokens.append(next_token_id)
            total_tokens_generated += 1

            if next_token_id in eos_token_ids:
                break

            # Forward pass
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

            # Phase-space dynamic monitoring
            loop_res = detector.step(next_token_id, last_hidden_state)

            # Baseline: Passive Early-Exit
            if passive_early_exit and loop_res.is_loop:
                rollback_history.append({
                    "type": "passive_early_exit",
                    "step": len(generated_tokens),
                    "reason": loop_res.reason,
                })
                break

            # DESPIN Tri-Regime Dynamic Trajectory Intervention
            if enable_despin and loop_res.is_loop:
                # REGIME 2: Post-Solution Overthinking (\\boxed{} already exists) -> PGEC Early Closure
                if enable_pgec and detector.has_boxed_answer:
                    rollback_history.append({
                        "type": "pgec_early_closure",
                        "at_token": len(generated_tokens),
                        "reason": f"Post-Solution Loop: {loop_res.reason} (Boxed=True)",
                    })
                    break

                # REGIME 3: Presentation Phase without boxed answer yet -> Pass-Through
                # (Model is formatting final steps towards \\boxed{}, do not execute destructive rollback)
                if not detector.in_think_phase:
                    continue

                # REGIME 1: Pre-Solution Impasse inside Deliberation Phase -> ASD Rollback
                if len(rollback_history) < self.max_rollbacks:
                    pivot_idx = max(0, loop_res.pivot_token_idx)
                    tokens_to_prune = len(generated_tokens) - pivot_idx

                    if tokens_to_prune > 0 and pivot_idx < len(generated_tokens):
                        target_kv_len = prompt_len + pivot_idx

                        # Physical O(1) Epistemic Slicing (KV Rollback)
                        past_key_values, slice_ms = KVCacheManager.rollback(
                            past_key_values, target_kv_len
                        )
                        total_rollback_ms += slice_ms

                        # Setup Attractor Subspace Deflation (ASD)
                        if loop_res.attractor_vector is not None:
                            u_attractor = loop_res.attractor_vector.to(self.device).to(self.W_u.dtype)
                            norm_u = torch.norm(u_attractor)
                            if norm_u > 1e-6:
                                u_unit = u_attractor / norm_u
                                proj = F.linear(u_unit.unsqueeze(0), self.W_u).squeeze(0)
                                if self.use_gram_schmidt:
                                    active_u_attractor = u_unit
                                    active_proj_Wu = F.relu(proj)
                                else:
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


# Aliases
ProductionEngine = DespinEngineV2
TriRegimeDespinEngine = DespinEngineV2
ProductionPhaseSpaceDetector = PhaseSpaceRecurrenceDetector

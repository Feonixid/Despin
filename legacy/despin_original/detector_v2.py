import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn.functional as F


@dataclass
class PhaseSpaceStep:
    step_id: int
    start_token_idx: int
    end_token_idx: int
    text: str
    tokens: List[int]
    hidden_state: torch.Tensor  # [hidden_dim]
    velocity: float = 1.0


@dataclass
class PhaseSpaceDetectionResult:
    is_loop: bool
    pivot_token_idx: int
    recurrence_sim: float
    loop_type: str = ""  # "limit_cycle", "velocity_collapse", "pgec_early_closure"
    reason: str = ""
    candidate_answer: Optional[str] = None
    attractor_vector: Optional[torch.Tensor] = None  # unit vector in hidden_dim


class PhaseSpaceRecurrenceDetector:
    """
    Phase-Space Recurrence & Tri-Regime Deliberation-Gated Detector:
    1. Distinguishes Deliberation Phase (<think>...</think>) from Presentation Phase.
    2. Warmup protection: first W tokens protected from premature intervention.
    3. Limit cycle detection: cos_sim >= 0.90 across non-adjacent steps.
    4. Angular velocity collapse: v < 0.18 across consecutive steps.
    5. Candidate Answer Tracking: specifically detects \\boxed{...} formulation.
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
        self.steps: List[PhaseSpaceStep] = []
        self.current_step_tokens: List[int] = []
        self.current_step_start_idx: int = 0
        self.total_tokens_seen: int = 0
        self.slow_step_streak: int = 0
        self.detected_loops_count: int = 0
        self.has_boxed_answer: bool = False
        self.in_think_phase: bool = True

    def is_step_boundary(self, token_id: int, current_text: str) -> bool:
        """Determines if a token corresponds to a logical step boundary."""
        token_str = self.tokenizer.decode([token_id])
        if "\n" in token_str or current_text.endswith((".", "!", "?", "...", ":\n")):
            return len(self.current_step_tokens) >= self.min_step_tokens
        return False

    def step(
        self,
        token_id: int,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> PhaseSpaceDetectionResult:
        """
        Process a single generated token and its hidden state online.
        """
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

        # Step boundary reached
        if hidden_state is None:
            h_vec = torch.zeros(1536)
        else:
            h_vec = hidden_state.detach().cpu().float().squeeze()

        # Step Angular Velocity: v_k = 1 - cos_sim(h_k, h_{k-1})
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

        # Reset step token buffer
        self.current_step_tokens = []
        self.current_step_start_idx = current_token_idx + 1

        if "\\boxed{" in step_text:
            self.has_boxed_answer = True

        # Warmup protection: protect first W tokens from false positive interventions
        if current_token_idx < self.warmup_tokens:
            return PhaseSpaceDetectionResult(
                is_loop=False, pivot_token_idx=-1, recurrence_sim=0.0
            )

        # Condition A: Limit Cycle Recurrence Scan
        best_sim = -1.0
        best_j = -1
        k = len(self.steps) - 1
        for j in range(k - 2):  # Scan non-adjacent historical steps
            sim = F.cosine_similarity(
                h_vec.unsqueeze(0), self.steps[j].hidden_state.unsqueeze(0), dim=-1
            ).item()
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
                reason=f"Recurrence: Step {k} ~ Step {best_j} (cos_sim={best_sim:.3f} >= {self.recurrence_threshold})",
                attractor_vector=u_attractor,
            )

        # Condition B: Angular Velocity Collapse
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
                reason=f"Velocity collapse: v={velocity:.4f} < {self.velocity_collapse_threshold} across {self.consecutive_slow_steps} steps",
                attractor_vector=u_attractor,
            )

        return PhaseSpaceDetectionResult(is_loop=False, pivot_token_idx=-1, recurrence_sim=0.0)

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set
import torch
import torch.nn.functional as F


@dataclass
class ReasoningStep:
    step_id: int
    start_token_idx: int
    end_token_idx: int
    text: str
    tokens: List[int]
    hidden_state: Optional[torch.Tensor] = None
    is_hesitation: bool = False
    repetition_score: float = 0.0
    candidate_answer: Optional[str] = None


@dataclass
class LoopDetectionResult:
    is_loop: bool
    pivot_token_idx: int
    loop_score: float
    loop_type: str = ""  # "impasse" or "overthinking_reverification"
    reason: str = ""
    candidate_answer: Optional[str] = None
    failed_tokens: List[int] = field(default_factory=list)


class AttractorDetector:
    """
    Online Streaming Attractor & Spinning-Wheel Detector.
    Monitors autoregressive reasoning sequences for:
    1. Impasse Loops: Circular doubt / repetitive restarts without reaching a candidate answer.
    2. Post-Derivation Overthinking: Redundant re-verification loops after an answer has already been derived.
    """

    HESITATION_PATTERNS = [
        r"(?i)\bwait\b",
        r"(?i)\blet me (re-?check|check|recalculate|re-evaluate|verify|re-read|rethink|re-examine)\b",
        r"(?i)\bhold on\b",
        r"(?i)\bis that right\??\b",
        r"(?i)\bactually,? (no|wait|let me|that may|maybe)\b",
        r"(?i)\blooking back\b",
        r"(?i)\bdoes this make sense\??\b",
        r"(?i)\balternatively,? wait\b",
        r"(?i)\bi made a mistake\b",
        r"(?i)\blet's try again\b",
        r"(?i)\bjust to (recap|double[- ]check|be sure)\b",
    ]

    CANDIDATE_ANSWER_PATTERNS = [
        r"(?i)(?:therefore|so|hence|thus),?\s*(?:the answer is|we have)?\s*([a-zA-Z]\s*=\s*[-+]?\d*\.?\d+(?:/\d+)?)",
        r"(?i)(?:therefore|so|hence|thus),?\s*(?:the answer is)?\s*([-+]?\d*\.?\d+(?:/\d+)?)\b",
        r"(?i)=\s*([-+]?\d*\.?\d+(?:/\d+)?)\s*$",
        r"\\boxed\{([^{}]+)\}",
    ]

    def __init__(
        self,
        tokenizer,
        semantic_sim_threshold: float = 0.88,
        ngram_repeat_threshold: float = 0.40,
        hesitation_burst_limit: int = 3,
        step_window_size: int = 6,
        min_step_tokens: int = 6,
    ):
        self.tokenizer = tokenizer
        self.semantic_sim_threshold = semantic_sim_threshold
        self.ngram_repeat_threshold = ngram_repeat_threshold
        self.hesitation_burst_limit = hesitation_burst_limit
        self.step_window_size = step_window_size
        self.min_step_tokens = min_step_tokens

        self.compiled_hesitations = [re.compile(p) for p in self.HESITATION_PATTERNS]
        self.compiled_candidates = [re.compile(p) for p in self.CANDIDATE_ANSWER_PATTERNS]
        self.reset()

    def reset(self):
        self.steps: List[ReasoningStep] = []
        self.current_step_tokens: List[int] = []
        self.current_step_start_idx: int = 0
        self.total_tokens_seen: int = 0
        self.consecutive_hesitations: int = 0
        self.detected_loops_count: int = 0
        self.latest_candidate_answer: Optional[str] = None
        self.candidate_answer_step_idx: Optional[int] = None

    def _extract_ngrams(self, text: str, n: int = 3) -> Set[str]:
        words = [w.strip(".,;:?!()[]{}'\"") for w in text.lower().split() if len(w) > 1]
        if len(words) < n:
            return set(words)
        return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

    def _compute_ngram_overlap(self, text_a: str, text_b: str, n: int = 3) -> float:
        ngrams_a = self._extract_ngrams(text_a, n)
        ngrams_b = self._extract_ngrams(text_b, n)
        if not ngrams_a or not ngrams_b:
            return 0.0
        intersection = ngrams_a.intersection(ngrams_b)
        union = ngrams_a.union(ngrams_b)
        return len(intersection) / float(len(union)) if union else 0.0

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
    ) -> LoopDetectionResult:
        """
        Process a single generated token online.
        """
        self.current_step_tokens.append(token_id)
        current_token_idx = self.total_tokens_seen
        self.total_tokens_seen += 1

        step_text = self.tokenizer.decode(self.current_step_tokens, skip_special_tokens=True)
        if not self.is_step_boundary(token_id, step_text):
            return LoopDetectionResult(is_loop=False, pivot_token_idx=-1, loop_score=0.0)

        # Check for candidate answer formulation in this step
        candidate_found = None
        for cand_pat in self.compiled_candidates:
            m = cand_pat.search(step_text)
            if m:
                candidate_found = m.group(1).strip()
                break

        if candidate_found:
            self.latest_candidate_answer = candidate_found
            self.candidate_answer_step_idx = len(self.steps)

        is_hesitation = any(pat.search(step_text) for pat in self.compiled_hesitations)

        # Repetition against recent steps
        max_repetition = 0.0
        recent_steps = self.steps[-self.step_window_size :]
        for prev_step in recent_steps:
            rep = self._compute_ngram_overlap(step_text, prev_step.text, n=3)
            if rep > max_repetition:
                max_repetition = rep

        # Semantic cosine similarity
        semantic_sim = 0.0
        if hidden_state is not None and recent_steps:
            last_step_hs = recent_steps[-1].hidden_state
            if last_step_hs is not None:
                curr_hs = hidden_state.detach().cpu().float().squeeze()
                prev_hs = last_step_hs.detach().cpu().float().squeeze()
                semantic_sim = float(
                    F.cosine_similarity(curr_hs, prev_hs, dim=-1).item()
                )

        new_step = ReasoningStep(
            step_id=len(self.steps),
            start_token_idx=self.current_step_start_idx,
            end_token_idx=current_token_idx,
            text=step_text,
            tokens=list(self.current_step_tokens),
            hidden_state=hidden_state.detach().cpu() if hidden_state is not None else None,
            is_hesitation=is_hesitation,
            repetition_score=max_repetition,
            candidate_answer=candidate_found,
        )
        self.steps.append(new_step)

        # Reset buffer
        self.current_step_tokens = []
        self.current_step_start_idx = current_token_idx + 1

        if is_hesitation:
            self.consecutive_hesitations += 1
        else:
            self.consecutive_hesitations = max(0, self.consecutive_hesitations - 1)

        # Classify: Is this an Overthinking Re-verification Loop or an Impasse Loop?
        has_derived_candidate = (
            self.latest_candidate_answer is not None
            and self.candidate_answer_step_idx is not None
            and (len(self.steps) - self.candidate_answer_step_idx) >= 1
        )

        # Condition 1: Overthinking after candidate answer is already formulated
        if has_derived_candidate and (self.consecutive_hesitations >= 2 or max_repetition >= 0.35):
            ans_step = self.steps[self.candidate_answer_step_idx]
            self.detected_loops_count += 1
            return LoopDetectionResult(
                is_loop=True,
                pivot_token_idx=ans_step.end_token_idx + 1,
                loop_score=max_repetition if max_repetition > 0.35 else 0.9,
                loop_type="overthinking_reverification",
                reason=f"Post-derivation overthinking loop detected (Candidate: {self.latest_candidate_answer})",
                candidate_answer=self.latest_candidate_answer,
                failed_tokens=new_step.tokens,
            )

        # Condition 2: High step n-gram repetition without candidate answer (Impasse)
        if max_repetition >= self.ngram_repeat_threshold:
            pivot_idx = new_step.start_token_idx
            self.detected_loops_count += 1
            return LoopDetectionResult(
                is_loop=True,
                pivot_token_idx=pivot_idx,
                loop_score=max_repetition,
                loop_type="impasse",
                reason=f"Impasse loop: repetitive steps ({max_repetition:.2f} >= {self.ngram_repeat_threshold})",
                candidate_answer=self.latest_candidate_answer,
                failed_tokens=new_step.tokens,
            )

        # Condition 3: Persistent Hesitation Burst without progress
        if self.consecutive_hesitations >= self.hesitation_burst_limit:
            burst_start_step = self.steps[-self.hesitation_burst_limit]
            pivot_idx = burst_start_step.start_token_idx
            self.detected_loops_count += 1
            loop_type = "overthinking_reverification" if has_derived_candidate else "impasse"
            return LoopDetectionResult(
                is_loop=True,
                pivot_token_idx=pivot_idx,
                loop_score=1.0,
                loop_type=loop_type,
                reason=f"Hesitation burst limit ({self.hesitation_burst_limit} hesitation steps)",
                candidate_answer=self.latest_candidate_answer,
                failed_tokens=burst_start_step.tokens,
            )

        # Condition 4: Semantic Stagnation
        if semantic_sim >= self.semantic_sim_threshold and is_hesitation:
            pivot_idx = new_step.start_token_idx
            self.detected_loops_count += 1
            return LoopDetectionResult(
                is_loop=True,
                pivot_token_idx=pivot_idx,
                loop_score=semantic_sim,
                loop_type="impasse",
                reason=f"Semantic stagnation ({semantic_sim:.2f} >= {self.semantic_sim_threshold})",
                candidate_answer=self.latest_candidate_answer,
                failed_tokens=new_step.tokens,
            )

        return LoopDetectionResult(is_loop=False, pivot_token_idx=-1, loop_score=0.0)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from despin.evaluator import MathEvaluator
import json

with open('results_v2/benchmark_v2_results.json', 'r') as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", dtype=torch.bfloat16, device_map="cuda")

from run_production_sweep import ProductionEngine, ProductionPhaseSpaceDetector

class AdaptivePhaseSpaceDetector(ProductionPhaseSpaceDetector):
    """
    Adaptive Phase-Space Detector with Scale-Invariant Recurrence Threshold:
    Short syntactic steps (L < 20 tokens, e.g. markdown headers) have high structural
    overlap, requiring tau = 0.93. Full reasoning steps (L >= 20 tokens) use tau = 0.90.
    """
    def step(self, token_id, hidden_state=None):
        self.current_step_tokens.append(token_id)
        current_token_idx = self.total_tokens_seen
        self.total_tokens_seen += 1

        step_text = self.tokenizer.decode(self.current_step_tokens, skip_special_tokens=True)
        if not self.is_step_boundary(token_id, step_text):
            from despin.detector_v2 import PhaseSpaceDetectionResult
            return PhaseSpaceDetectionResult(is_loop=False, pivot_token_idx=-1, recurrence_sim=0.0)

        h_vec = hidden_state.detach().cpu().float().squeeze() if hidden_state is not None else torch.zeros(1536)

        velocity = 1.0
        if self.steps:
            prev_h = self.steps[-1].hidden_state
            cos_prev = torch.nn.functional.cosine_similarity(h_vec.unsqueeze(0), prev_h.unsqueeze(0), dim=-1).item()
            velocity = 1.0 - cos_prev

        from despin.detector_v2 import PhaseSpaceStep, PhaseSpaceDetectionResult
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
        step_len = len(self.current_step_tokens)
        self.current_step_tokens = []
        self.current_step_start_idx = current_token_idx + 1

        if "\\boxed{" in step_text or "the answer is" in step_text.lower():
            self.candidate_answer_step_idx = len(self.steps) - 1

        # Warmup protection
        if current_token_idx < self.warmup_tokens:
            return PhaseSpaceDetectionResult(is_loop=False, pivot_token_idx=-1, recurrence_sim=0.0)

        # Scale-invariant threshold: short steps require higher similarity to declare an attractor
        eff_threshold = 0.93 if step_len < 20 else self.recurrence_threshold

        best_sim = -1.0
        best_j = -1
        k = len(self.steps) - 1
        for j in range(k - 2):
            sim = torch.nn.functional.cosine_similarity(h_vec.unsqueeze(0), self.steps[j].hidden_state.unsqueeze(0), dim=-1).item()
            if sim > best_sim:
                best_sim = sim
                best_j = j

        if best_sim >= eff_threshold:
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
                reason=f"Adaptive limit cycle: Step {k} ~ Step {best_j} (sim={best_sim:.3f} >= {eff_threshold}, len={step_len})",
                attractor_vector=u_attractor,
            )

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

engine = ProductionEngine(
    model=model,
    tokenizer=tokenizer,
    recurrence_threshold=0.90,
    velocity_collapse_threshold=0.18,
    warmup_tokens=80,
    deflation_gamma=15.0,
    exploration_steps=6,
)

# Test on math500_1 (the hard rescue) and gsm8k_0 (the clean problem)
test_ids = ['math500_1', 'gsm8k_0', 'gsm8k_4']
test_probs = [r['problem'] for r in data['records'] if r['problem']['id'] in test_ids]

import run_production_sweep
run_production_sweep.ProductionPhaseSpaceDetector = AdaptivePhaseSpaceDetector

print("=== Running Adaptive PSRD Validation Test ===")
for p in test_probs:
    messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {p['question']}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    res = engine.generate(prompt=prompt, max_new_tokens=2048, enable_pgec=True)
    pred = MathEvaluator.extract_answer(res.text)
    corr = MathEvaluator.is_equiv(pred, p['ground_truth'])
    print(f"{p['id']:10s} | Correct={str(corr):5s} | Pred={str(pred)[:15]:15s} | Tok={res.total_tokens_generated:4d} | Actions={len(res.rollback_history)}")


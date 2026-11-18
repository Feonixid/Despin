import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from despin.evaluator import MathEvaluator
from run_master_benchmark import MasterPhaseSpaceDetector, MasterDespinEngine

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")

with open('results_v2/benchmark_v2_results.json', 'r') as f:
    import json
    data = json.load(f)

prob = next(r['problem'] for r in data['records'] if r['problem']['id'] == 'math500_1')
q = prob['question']
gt = prob['ground_truth']
messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Here we test recurrence_threshold = 0.90 without the velocity < 0.35 restriction
class CleanDetector(MasterPhaseSpaceDetector):
    def step(self, token_id, hidden_state=None):
        res = super().step(token_id, hidden_state)
        return res

# Let's override the limit cycle condition to be pure recurrence >= 0.90
class ExactCalibratedDetector(MasterPhaseSpaceDetector):
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
        self.current_step_tokens = []
        self.current_step_start_idx = current_token_idx + 1

        if "\\boxed{" in step_text or "the answer is" in step_text.lower():
            self.candidate_answer_step_idx = len(self.steps) - 1

        # Warmup protection: first 80 tokens
        if current_token_idx < self.warmup_tokens:
            return PhaseSpaceDetectionResult(is_loop=False, pivot_token_idx=-1, recurrence_sim=0.0)

        best_sim = -1.0
        best_j = -1
        k = len(self.steps) - 1
        for j in range(k - 2):
            sim = torch.nn.functional.cosine_similarity(h_vec.unsqueeze(0), self.steps[j].hidden_state.unsqueeze(0), dim=-1).item()
            if sim > best_sim:
                best_sim = sim
                best_j = j

        # Pure recurrence condition at 0.90 (after warmup)
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

engine = MasterDespinEngine(
    model=model,
    tokenizer=tokenizer,
    recurrence_threshold=0.90,
    velocity_collapse_threshold=0.18,
    warmup_tokens=80,
    deflation_gamma=15.0,
    exploration_steps=6,
)
# Monkey patch detector class
import run_master_benchmark
run_master_benchmark.MasterPhaseSpaceDetector = ExactCalibratedDetector

res = engine.generate(prompt=prompt, max_new_tokens=2048, enable_pgec=True)
pred = MathEvaluator.extract_answer(res.text)
correct = MathEvaluator.is_equiv(pred, gt)
print(f"math500_1 Test: Correct={correct} | Pred={pred} | GT={gt} | Tokens={res.total_tokens_generated} | Pruned={res.pruned_tokens_count}")


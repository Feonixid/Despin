import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from despin.engine import DespinEngine
from despin.evaluator import MathEvaluator

def test_pipeline():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    engine = DespinEngine(model=model, tokenizer=tokenizer)

    question = "If x + 2y = 10 and 3x - y = 9, what is the value of x * y?"
    messages = [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {question}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("\n--- 1. Running Greedy Baseline ---")
    res_greedy = engine.generate(
        prompt=prompt,
        max_new_tokens=1024,
        base_temperature=0.0,
        enable_despin=False,
    )
    ans_greedy = MathEvaluator.extract_answer(res_greedy.text)
    print(f"Greedy: {res_greedy.final_tokens_count} tokens in {res_greedy.generation_latency_s:.2f}s | Answer: {ans_greedy}")

    print("\n--- 2. Running Passive Early-Exit Baseline (Wei et al. 2025) ---")
    res_passive = engine.generate(
        prompt=prompt,
        max_new_tokens=1024,
        base_temperature=0.0,
        enable_despin=False,
        passive_early_exit=True,
    )
    ans_passive = MathEvaluator.extract_answer(res_passive.text)
    print(f"Passive Early-Exit: {res_passive.final_tokens_count} tokens in {res_passive.generation_latency_s:.2f}s | Answer: {ans_passive}")

    print("\n--- 3. Running DESPIN (Our Method) ---")
    res_despin = engine.generate(
        prompt=prompt,
        max_new_tokens=1024,
        base_temperature=0.0,
        enable_despin=True,
    )
    ans_despin = MathEvaluator.extract_answer(res_despin.text)
    print(f"DESPIN: {res_despin.final_tokens_count} tokens, Pruned: {res_despin.pruned_tokens_count}, Rollbacks: {res_despin.rollbacks_count} in {res_despin.generation_latency_s:.2f}s | Answer: {ans_despin}")
    print(f"Rollback History: {res_despin.rollback_history}")

    gt = "12"
    print(f"\nGround Truth: {gt}")
    print(f"Greedy Correct: {MathEvaluator.is_equiv(ans_greedy, gt)}")
    print(f"Passive Correct: {MathEvaluator.is_equiv(ans_passive, gt)}")
    print(f"DESPIN Correct: {MathEvaluator.is_equiv(ans_despin, gt)}")

if __name__ == "__main__":
    test_pipeline()

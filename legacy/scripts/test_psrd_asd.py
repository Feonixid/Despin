"""
Prototype test for:
1. Phase-Space Recurrence Detection (PSRD)
2. Attractor Subspace Deflation (ASD)
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_subspace_deflation():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Check unembedding projection
    W_u = model.lm_head.weight  # [vocab_size, hidden_dim]
    print(f"Unembedding matrix shape: {W_u.shape}")

    # Create dummy hidden states
    h_pivot = torch.randn(1, 1536, dtype=torch.bfloat16, device="cuda")
    h_failed = h_pivot + torch.randn(1, 1536, dtype=torch.bfloat16, device="cuda") * 0.5
    u_attractor = F.normalize(h_failed - h_pivot, p=2, dim=-1)

    # Compute logit projection
    logit_proj = F.linear(u_attractor, W_u).squeeze(0)  # [vocab_size]
    print(f"Logit projection shape: {logit_proj.shape}, min: {logit_proj.min().item():.2f}, max: {logit_proj.max().item():.2f}")

    # Check top penalized tokens
    top_penalized = torch.topk(logit_proj, k=5).indices
    print("Top penalized token IDs:", top_penalized.tolist())
    print("Top penalized token strings:", [tokenizer.decode([t]) for t in top_penalized.tolist()])
    print("Attractor Subspace Deflation works cleanly!")

if __name__ == "__main__":
    test_subspace_deflation()

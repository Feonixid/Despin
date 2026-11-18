from despin.detector_v2 import PhaseSpaceRecurrenceDetector
from transformers import AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
d = PhaseSpaceRecurrenceDetector(tok)
d.step(10, torch.randn(1536))
print("Hidden state norm:", torch.norm(d.steps[0].hidden_state).item() if d.steps else "No step yet")


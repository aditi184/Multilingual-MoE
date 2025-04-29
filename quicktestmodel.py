from olmo.model import OLMo
import torch
model = OLMo.from_checkpoint("/home/mila/k/khandela/scratch/ai2-llm/checkpoints/OLMoE/latest_pretrained_olmo_format")
model = model.cuda()
model = model.to(torch.bfloat16)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
inputs = tokenizer("Bitcoin is", return_tensors="pt")
inputs = {k: v.cuda() for k, v in inputs.items()}
out = model.generate(**inputs)
print(tokenizer.decode(out[0][0][0]))
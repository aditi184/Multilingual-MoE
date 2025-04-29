
# -*- coding: utf-8 -*-:
# from olmo.model import OLMo
# import torch
# model = OLMo.from_checkpoint("/home/mila/k/khandela/scratch/ai2-llm/runs/pretrain-hi-en/latest-unsharded")
# # model = OLMo.from_checkpoint("/home/mila/k/khandela/scratch/ai2-llm/checkpoints/OLMoE/base-0924")
# print(model)
# # model.print_trainable_parameters()
# model = model.cuda()
# model = model.to()
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")
# inputs = tokenizer("Bitcoin is", return_tensors="pt")
# inputs = {k: v.cuda() for k, v in inputs.items()}
# out = model.generate(**inputs)
# print(tokenizer.decode(out[0][0][0]))

# inputs = tokenizer("भारत एक", return_tensors="pt")
# inputs = {k: v.cuda() for k, v in inputs.items()}
# out = model.generate(**inputs)
# print(tokenizer.decode(out[0][0][0]))



from transformers import OlmoeForCausalLM, AutoTokenizer
import torch
model = OlmoeForCausalLM.from_pretrained("/home/mila/k/khandela/scratch/ai2-llm/checkpoints/OLMoE/multilingual-en-hi-ar-ru-zh-700", torch_dtype=torch.bfloat16).cuda()
# model = OlmoeForCausalLM.from_pretrained("/home/mila/k/khandela/scratch/ai2-llm/checkpoints/OLMoE/multilingual-en-hi-ar-ru-zh-700").cuda()
# print(model)

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")
inputs = tokenizer("नमस्ते", return_tensors="pt")
inputs = {k: v.cuda() for k, v in inputs.items()}
out = model.generate(**inputs, max_length=64)
print(tokenizer.decode(out[0]))
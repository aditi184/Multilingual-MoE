
# # -*- coding: utf-8 -*-:
# from olmo.model import OLMo
import torch
# model = OLMo.from_checkpoint("/home/mila/k/khandela/scratch/ai2-llm/runs/pretrain-multilingual-low-resource/step2300")
# # # model = OLMo.from_checkpoint("/home/mila/k/khandela/scratch/ai2-llm/checkpoints/OLMoE/base-0924")
# print(model)
# # # model.print_trainable_parameters()
# model = model.cuda()
# # model = model.to()
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")
# inputs = tokenizer("दृश्यात्मक स्पेक्ट्रममा अवस्थित नभएको नयाँ रङको वर्णन गर्नुहोस्।", return_tensors="pt")
# inputs = {k: v.cuda() for k, v in inputs.items()}
# out = model.generate(**inputs)
# print(tokenizer.decode(out[0][0][0]))

# inputs = tokenizer("भारत एक", return_tensors="pt")
# inputs = {k: v.cuda() for k, v in inputs.items()}
# out = model.generate(**inputs)
# print(tokenizer.decode(out[0][0][0]))



from transformers import OlmoeForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# model = AutoModelForCausalLM.from_pretrained("/home/mila/k/khandela/scratch/ai2-llm/checkpoints/OLMo-1B")
model = OlmoeForCausalLM.from_pretrained("/home/mila/k/khandela/scratch/ai2-llm/checkpoints/OLMoE/retrain-5lang",torch_dtype=torch.bfloat16).cuda()
# # model = OlmoeForCausalLM.from_pretrained("/home/mila/k/khandela/scratch/ai2-llm/checkpoints/OLMoE/multilingual-en-hi-ar-ru-zh-700").cuda()
print(model)

# from transformers import AutoModelForCausalLM, AutoTokenizer
# olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
# tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
# print(olmo)
# print(tokenizer)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")
inputs = tokenizer("नमस्ते", return_tensors="pt")
inputs = {k: v.cuda() for k, v in inputs.items()}
out = model.generate(**inputs, max_length=64)
print(tokenizer.decode(out[0]))
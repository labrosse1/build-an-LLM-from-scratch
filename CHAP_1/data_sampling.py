import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader 


tokenizer = tiktoken.get_encoding("gpt2")

with open("the-verdict.txt", "r") as file:
    raw_text = file.read()


encoded_text = tokenizer.encode(raw_text)
print(len(encoded_text))
encoded_sample = encoded_text[50:]

context_size = 4
x = encoded_sample[:context_size]
y = encoded_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

'''
print(raw_text[:context_size])
print(raw_text[1:context_size+1])
'''

for i in range(1, context_size+1):
    context = encoded_sample[:i]
    desired = encoded_sample[i]
    print(context, "---->", tokenizer.decode([desired]))



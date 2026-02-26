import torch
from model import GPT
import tiktoken

model = GPT.from_pretrained()

model.eval()

enc = tiktoken.get_encoding("gpt2")
input_text = "Once upon a time, there was a boy who"
tokens = torch.tensor(enc.encode(input_text))
tokens = tokens.unsqueeze(0)
output = model.generate(tokens, max_new_tokens=256)
print(enc.decode(output[0].tolist()))

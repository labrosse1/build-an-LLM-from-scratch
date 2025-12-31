# NEW: use CPU here as inference is cheap with 
# this model and to ensure readers get same results in the
# remaining sections of this book

import torch
from CHAP_3.GPT_backbone import model
from CHAP_3.GPT_backbone import generate_text_simple
from CHAP_4.Pretraining_unlabeled_data import text_to_token_ids
from CHAP_4.Pretraining_unlabeled_data import token_ids_to_text
from CHAP_4.Pretraining_unlabeled_data import GPT_CONFIG_124M
import tiktoken

inference_device = torch.device("cpu")

model.to(inference_device)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
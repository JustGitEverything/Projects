import numpy as np
import cupy as cp
from GPT import GPT
import process_tokens
import pickle
import json

with open(r"""C:\Users\Justin Hohenstein\PycharmProjects\datasets\fineData\training_unique_chars.json""", "r", encoding="utf-8") as fh:
    loaded_chars = json.load(fh)

print('chars', loaded_chars)

chars = len(loaded_chars)
print("char length", chars)

cti = process_tokens.load_cti()
itc = process_tokens.load_itc()

vocab_size = len(cti)
print("Vocabulary Size:", vocab_size)

n_embd = 768
block_size = 128
num_heads = 8
weight_decay = 1e-3


initial_lr = 0.0002

# gpt = GPT(vocab_size, n_embd, block_size, num_heads, drop_rate=0.1, n_layer=8, lr=initial_lr, weight_decay=weight_decay)
gpt = pickle.load(open("gpt.pickle", "rb"))
print("gpt param count:", gpt.p_count())

shard_pieces = 10
large_steps = 50

text = "The US is "
text = process_tokens.encode(text, cti)
text = process_tokens.priority_merge(text, cti)

print("encoded text", text)

print(process_tokens.decode(process_tokens.ret(gpt.generate(cp.array(text, dtype=cp.int32), 100).tolist(), itc), itc))
print(process_tokens.decode(process_tokens.ret(gpt.generate(cp.array(text, dtype=cp.int32), 100).tolist(), itc), itc))
print(process_tokens.decode(process_tokens.ret(gpt.generate(cp.array(text, dtype=cp.int32), 100).tolist(), itc), itc))

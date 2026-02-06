import numpy as np
import cupy as cp
from GPT import GPT
import tiktoken

tokenizer = tiktoken.get_encoding("p50k_base")

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = 50257

print("characters", chars)
print("Vocabulary Size:", vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = []
for i in range(len(text) // 2000):
    data.extend(tokenizer.encode(text[i * 2000: (i + 1) * 2000]))

data = np.array(data)

print("data", data.shape)

n = int(data.shape[0] * 0.9)
train_data = cp.array(data[:n])
test_data = cp.array(data[n:])

n_embd = 768
block_size = 128
num_heads = 8
weight_decay = 1e-3

batch_size = 8
its_per_batch = 8
iterations = int(batch_size / its_per_batch)


def get_batch(split):
    dt = train_data if split == 'train' else test_data
    idx = cp.random.randint(0, len(dt) - block_size, size=its_per_batch)

    # gather using broadcasting
    offsets = cp.arange(block_size)
    bx = dt[idx[:, None] + offsets[None, :]]
    by = dt[idx[:, None] + offsets[None, :] + 1]

    return bx, by


gpt = GPT(vocab_size, n_embd, block_size, num_heads, drop_rate=0.1, n_layer=8, lr=0.0002, weight_decay=weight_decay)
print("gpt param count:", gpt.p_count())

for e in range(4001):
    loss = 0
    for i in range(iterations):
        x, y = get_batch('train')

        y_hat = gpt.forward(x, training=True)
        loss += gpt.backward(y)

    print("loss", e, cp.average(loss) / iterations, "norm", gpt.grad_norm(batch_size))
    gpt.update(batch_size, e)

    if e % 200 == 0:
        eval_loss = 0
        for i in range(its_per_batch):
            x, y = get_batch('eval')

            y_hat = gpt.forward(x, training=False)

            _, nl = gpt.head.backward(y)
            gpt.head.reset()

            eval_loss += nl

        print("evaluation loss:", cp.average(eval_loss) / its_per_batch)
        print(tokenizer.decode(gpt.generate(cp.array([198] if e % 400 == 0 else [3237, 18029, 25], dtype=cp.int32), 126).tolist()))

print(tokenizer.decode(gpt.generate(cp.array([3237, 18029, 25], dtype=cp.int32), 120).tolist()))
print(tokenizer.decode(gpt.generate(cp.array([198], dtype=cp.int32), 126).tolist()))
print(tokenizer.decode(gpt.generate(cp.array([198], dtype=cp.int32), 126).tolist()))

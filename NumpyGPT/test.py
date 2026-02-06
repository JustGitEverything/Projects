import numpy as np
import cupy as cp
from GPT import GPT

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print("characters", chars)
print("Vocabulary Size:", vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = np.array(encode(text), dtype=np.int32)

n = int(data.shape[0] * 0.9)
train_data = data[:n]
test_data = data[n:]

n_embd = 80
block_size = 128
num_heads = 4
weight_decay = 1e-2

batch_size = 64
its_per_batch = 64
iterations = int(batch_size / its_per_batch)


def get_batch(split):
    dt = train_data if split == 'train' else test_data

    indices = np.random.randint(len(dt) - block_size, size=its_per_batch)

    bx = np.array([dt[ix:ix + block_size] for ix in indices])
    by = np.array([dt[ix + 1: ix + block_size + 1] for ix in indices])

    return bx, by


gpt = GPT(vocab_size, n_embd, block_size, num_heads, drop_rate=0.2, n_layer=12, lr=0.01, weight_decay=weight_decay)
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
        print(decode(gpt.generate(cp.array([0], dtype=cp.int32), 126).tolist()))

print(decode(gpt.generate(cp.array([0], dtype=cp.int32), 126).tolist()))
print(decode(gpt.generate(cp.array([0], dtype=cp.int32), 126).tolist()))
print(decode(gpt.generate(cp.array([0], dtype=cp.int32), 126).tolist()))

import numpy as np
import cupy as cp
from GPT import GPT
import process_tokens
import pickle

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = len(sorted(list(set(text))))
print("char length", chars)

cti = process_tokens.load_cti()
itc = process_tokens.load_itc()

vocab_size = len(cti)
print("Vocabulary Size:", vocab_size)

text = process_tokens.encode(text, cti)
print("text encoding done")

print("length text pre", len(text))
text = process_tokens.quick_merge(text, cti)  # [:200000]
# print("ret", process_tokens.decode(process_tokens.ret(text, itc, chars), itc))
# print(process_tokens.easy_colors(text, itc, chars))

data = np.array(text)
# print("D", data)

print("length data post", data.shape)

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
# gpt = pickle.load(open("gpt.pickle", "rb"))
print("gpt param count:", gpt.p_count())

for e in range(20001):
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
        print(process_tokens.decode(process_tokens.rtn(gpt.generate(cp.array([0] if e % 400 == 0 else [1], dtype=cp.int32), 120).tolist(), itc, chars), itc))

print(process_tokens.decode(process_tokens.ret(gpt.generate(cp.array([0], dtype=cp.int32), 120).tolist(), itc, chars), itc))
print(process_tokens.decode(process_tokens.ret(gpt.generate(cp.array([0], dtype=cp.int32), 120).tolist(), itc, chars), itc))
print(process_tokens.decode(process_tokens.ret(gpt.generate(cp.array([0], dtype=cp.int32), 120).tolist(), itc, chars), itc))

# Save the model
pickle.dump(gpt, file=open("gpt.pickle", "wb"))

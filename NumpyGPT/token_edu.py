import numpy as np
import cupy as cp
from GPT import GPT
import process_tokens
import util
import pickle
import json

with open(r"""C:\Users\Justin Hohenstein\PycharmProjects\datasets\fineData\training_unique_chars.json""", "r", encoding="utf-8") as fh:
    loaded_chars = json.load(fh)

print('chars', loaded_chars)
print("char length", len(loaded_chars))

cti = process_tokens.load_cti()
itc = process_tokens.load_itc()
# print(itc)

vocab_size = len(cti)
print("Vocabulary Size:", vocab_size)

n_embd = 768
block_size = 128
num_heads = 8
weight_decay = 1e-3

batch_size = 16


def get_batch(dt, step):
    start = step * batch_size * block_size
    end = start + batch_size * block_size

    bx = dt[start:end].reshape(batch_size, block_size)
    by = dt[start + 1:end + 1].reshape(batch_size, block_size)

    return bx, by


initial_lr = 0.0002

#  gpt = GPT(vocab_size, n_embd, block_size, num_heads, drop_rate=0.1, n_layer=8, lr=initial_lr, weight_decay=weight_decay)
gpt = pickle.load(open("gpt.pickle", "rb"))
gpt.set_lr(0.0002)
print("gpt param count:", gpt.p_count())

shard_pieces = 10
large_steps = 500000

for e in range(1):
    for shard in range(10):
        with open(r"""C:\Users\Justin Hohenstein\PycharmProjects\datasets\fineData\shard_0""" + str(shard) + ".txt",
                  'r', encoding='utf-8') as f:
            text = f.read()

        text = process_tokens.encode(text, cti)
        print("Shard", shard, "encoding done")

        print("length text pre", len(text))
        text = process_tokens.priority_merge(text, cti)
        # print("ret", process_tokens.decode(process_tokens.ret(text, itc, chars), itc))
        # print(process_tokens.easy_colors(text, itc, chars))

        data = np.array(text)
        # print("D", data)

        print("length text post", data.shape)

        data_per_shard = data.shape[0] // shard_pieces

        for shard_piece in range(shard_pieces):
            if e < 1 and shard * shard_pieces + shard_piece <= large_steps:
                new_lr = initial_lr * (0.1 + 0.9 * 0.5 *
                                       (1 + cp.cos(cp.pi * (shard * shard_pieces + shard_piece) / large_steps)))

                # gpt.set_lr(new_lr)
                print("set learning rate to", new_lr)

            # training shard
            print("Shard", shard, "piece", shard_piece + 1, "/", shard_pieces)
            gpu_data = data[shard_piece * data_per_shard:(shard_piece + 1) * data_per_shard]
            n = int(gpu_data.shape[0] * 0.9)
            train_data = cp.array(gpu_data[:n])
            test_data = cp.array(gpu_data[n:])

            its_per_shard_piece = (n - 1) // (batch_size * block_size)

            for i in range(its_per_shard_piece):
                x, y = get_batch(train_data, i)
                y_hat = gpt.forward(x, training=True)
                loss = gpt.backward(y)

                print("Shard", shard, "piece", shard_piece + 1, "/", shard_pieces, "i", i + 1, "/", its_per_shard_piece,
                      "loss:", cp.average(loss), "norm", gpt.grad_norm(batch_size))
                # gpt.update(batch_size, shard * shard_pieces * its_per_shard_piece + shard_piece * its_per_shard_piece + i)
                gpt.update(batch_size, 10000)

            # eval
            x, y = get_batch(test_data, 0)
            y_hat = gpt.forward(x, training=False)
            nl = util.CrossEntropy.a(y_hat, y)

            print("Shard", shard, "piece", shard_piece + 1, "/", shard_pieces, "eval loss:", cp.average(nl))

            print(process_tokens.decode(process_tokens.ret(gpt.generate(cp.array(process_tokens.priority_merge(process_tokens.encode("Hello, I am a large language model and ", cti), cti), dtype=cp.int32), 100).tolist(), itc), itc))
            print(process_tokens.decode(process_tokens.ret(gpt.generate(cp.array(process_tokens.priority_merge(process_tokens.encode("Taylor Swift is ", cti), cti), dtype=cp.int32), 100).tolist(), itc), itc))

print(process_tokens.decode(process_tokens.ret(gpt.generate(cp.array(process_tokens.priority_merge(process_tokens.encode("Hello, I am a large language model and ", cti), cti), dtype=cp.int32), 100).tolist(), itc), itc))
print(process_tokens.decode(process_tokens.ret(gpt.generate(cp.array([0], dtype=cp.int32), 120).tolist(), itc), itc))
print(process_tokens.decode(process_tokens.ret(gpt.generate(cp.array([0], dtype=cp.int32), 120).tolist(), itc), itc))
print(process_tokens.decode(process_tokens.ret(gpt.generate(cp.array([0], dtype=cp.int32), 120).tolist(), itc), itc))

# Save the model
pickle.dump(gpt, file=open("gpt.pickle", "wb"))

import cupy as np

import util
from Head import Head
from Layer import Layer
from LayerNorm import LayerNorm
from Dropout import Dropout


class Block:
    def __init__(self, num_embed, block_size, num_heads, drop_rate=0, lr=0.01, weight_decay=1e-1):
        self.num_embed = num_embed
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_size = int(num_embed / num_heads)

        self.ln1 = LayerNorm(num_embed, lr=lr)
        self.heads = [Head(num_embed, self.head_size, block_size, drop_rate=drop_rate, lr=lr, weight_decay=weight_decay)
                      for _ in range(num_heads)]
        self.proj = Layer(num_embed, num_embed, util.N, lr=lr, weight_decay=weight_decay)
        self.dropout1 = Dropout(num_embed, drop_rate=drop_rate)

        self.ln2 = LayerNorm(num_embed, lr=lr)
        self.la2 = Layer(num_embed, num_embed * 4, util.ReLU, lr=lr, weight_decay=weight_decay)
        self.la3 = Layer(num_embed * 4, num_embed, util.N, lr=lr, weight_decay=weight_decay)
        self.dropout2 = Dropout(num_embed, drop_rate=drop_rate)

    def forward(self, x, block_size=None, training=False):
        res_1 = x
        # print("WHY", x)
        x = self.ln1.forward(x)
        # print("BF", x)
        # print("PREV", x)
        x = np.concatenate([head.forward(x, block_size=block_size, training=training) for head in self.heads], axis=-1)
        # print("POST", x)
        # print("FR", x.shape, x)
        x = self.proj.forward(x)

        x = res_1 + self.dropout1.forward(x, training=training)

        res_2 = x
        x = self.ln2.forward(x)
        x = self.la2.forward(x)
        x = self.la3.forward(x)
        x = res_2 + self.dropout2.forward(x, training=training)

        return x

    def p_count(self):
        p_count = self.ln1.p_count()
        for head in self.heads:
            p_count += head.p_count()
        return p_count + self.proj.p_count() + self.ln2.p_count() + self.la2.p_count() + self.la3.p_count()

    def grad_sq(self, gi):
        p_count = self.ln1.grad_sq(gi)
        for head in self.heads:
            p_count += head.grad_sq(gi)
        return p_count + self.proj.grad_sq(gi) + self.ln2.grad_sq(gi) + self.la2.grad_sq(gi) + self.la3.grad_sq(gi)

    def set_lr(self, lr):
        self.ln1.set_lr(lr)

        for head in self.heads:
            head.set_lr(lr)

        self.proj.set_lr(lr)
        self.ln2.set_lr(lr)
        self.la2.set_lr(lr)
        self.la3.set_lr(lr)

    def backward(self, inner_d):
        dres_2 = inner_d
        inner_d = self.dropout2.backward(inner_d)
        inner_d = self.la3.backward(inner_d)
        inner_d = self.la2.backward(inner_d)
        inner_d = self.ln2.backward(inner_d)

        inner_d = dres_2 + inner_d

        dres_1 = inner_d
        inner_d = self.dropout1.backward(inner_d)
        inner_d = self.proj.backward(inner_d)
        inner_ds = np.split(inner_d, self.num_heads, axis=-1)
        # print("IDS", inner_ds)
        inner_d = np.sum(np.array([head.backward(in_d) for head, in_d in zip(self.heads, inner_ds)]), axis=0)
        # print("INNER", inner_d)
        # print("ORG", [head.backward(in_d) for head, in_d in zip(self.heads, inner_ds)])
        inner_d = self.ln1.backward(inner_d)

        inner_d = dres_1 + inner_d

        return inner_d

    def update(self, its, t):
        self.ln1.update(its, t)

        for head in self.heads:
            head.update(its, t)

        self.proj.update(its, t)
        self.ln2.update(its, t)
        self.la2.update(its, t)
        self.la3.update(its, t)

import cupy as np
import util

from Encode import Encode
from Dropout import Dropout


class Head:
    def __init__(self, n_embd, n_head, block_size, lr=0.01, drop_rate=0, weight_decay=1e-1):
        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size

        self.query = Encode(n_embd, n_head, lr=lr, weight_decay=weight_decay)
        self.key = Encode(n_embd, n_head, lr=lr, weight_decay=weight_decay)
        self.value = Encode(n_embd, n_head, lr=lr, weight_decay=weight_decay)

        self.dropout = Dropout(n_embd, drop_rate)

        self.q = []
        self.k = []
        self.v = []

        self.attn = []
        self.wei = []

        self.buffer = np.tril(np.ones((block_size, block_size)))

    def forward(self, x, block_size=None, training=False):
        block_size = self.block_size if block_size is None else block_size
        # print("HELP", x)
        # print("X", x)

        self.q = self.query.forward(x)
        self.k = self.key.forward(x)
        self.v = self.value.forward(x)

        # print("Q", self.q.shape)

        # print(np.sqrt(self.n_head))
        # print("OK NOW THOUGH", self.q, self.k, np.matmul(self.q, self.k.transpose(0, 2, 1)))

        wei = np.matmul(self.q, self.k.transpose(0, 2, 1)) / np.sqrt(self.n_head)
        # print("wei", wei)
        wei[:, self.buffer[:block_size, :block_size] == 0] = - np.inf
        # print("MASKED", wei)
        # print("tri", wei)
        # print("MUST BE HERE", wei)
        wei = util.Softmax.a(wei)
        # print("AND AFTER", wei)
        # print("sm", wei)

        self.attn = wei

        wei = self.dropout.forward(wei, training=training)

        # print("MM", wei.shape, self.v.shape)

        y = np.matmul(wei, self.v)

        self.wei = wei

        return y

    def backward(self, inner_d):
        # print("ID", inner_d)
        # print("WEI DIM", self.wei.shape, "inner d", inner_d.shape)
        v_d = np.matmul(self.wei.transpose(0, 2, 1), inner_d)
        v_inner = self.value.backward(v_d)
        # print("V DIM", self.v.shape, "inner d", inner_d.shape)

        wei_d = np.matmul(inner_d, self.v.transpose(0, 2, 1))
        wei_d = self.dropout.backward(wei_d)
        wei_inner = util.Softmax.a_p(self.attn, wei_d)
        # print("WI", wei_inner)
        # print("BUF", self.buffer[:self.block_size, :self.block_size])
        # wei_inner = wei_inner * self.buffer[:self.block_size, :self.block_size]

        wei_inner = wei_inner / np.sqrt(self.n_head)

        k_d = np.matmul(wei_inner.transpose(0, 2, 1), self.q)
        k_inner = self.key.backward(k_d)

        q_d = np.matmul(wei_inner, self.k)
        q_inner = self.query.backward(q_d)

        inner_d = v_inner + k_inner + q_inner

        return inner_d

    def p_count(self):
        return self.query.p_count() + self.key.p_count() + self.value.p_count()

    def grad_sq(self, gi):
        return self.query.grad_sq(gi) + self.key.grad_sq(gi) + self.value.grad_sq(gi)

    def set_lr(self, lr):
        self.query.set_lr(lr)
        self.key.set_lr(lr)
        self.value.set_lr(lr)

    def update(self, its, t):
        self.query.update(its, t)
        self.key.update(its, t)
        self.value.update(its, t)

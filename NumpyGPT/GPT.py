import cupy as np
from Embedding import Embedding
from TiedLayer import TiedLayer
from LayerNorm import LayerNorm
from Block import Block


class GPT:
    def __init__(self, vocab_size, num_embd, block_size, num_heads, n_layer=6, drop_rate=0, lr=0.01, weight_decay=1e-1):
        self.vocab_size = vocab_size
        self.num_embd = num_embd
        self.block_size = block_size

        self.lr = lr

        self.token_embedding = Embedding(vocab_size, num_embd, lr=lr, weight_decay=weight_decay)
        self.positional_embedding = Embedding(block_size, num_embd, lr=lr, weight_decay=weight_decay)

        self.blocks = [Block(num_embd, block_size, num_heads, drop_rate=drop_rate, lr=lr, weight_decay=weight_decay)
                       for _ in range(n_layer)]

        self.ln = LayerNorm(num_embd, lr=lr)

        self.head = TiedLayer(num_embd, vocab_size, self.token_embedding, lr=lr, weight_decay=weight_decay)

        # print("se", self.token_embedding.weights.shape, self.head.weights.shape)

    def forward(self, x, block_size=None, training=False):
        tok_emb = self.token_embedding.forward(x)
        pos_emb = self.positional_embedding.forward(np.arange(self.block_size if block_size is None else block_size))

        # print("TK", tok_emb)
        # print("PE", pos_emb)
        embedding = tok_emb + pos_emb

        # print("EMB", embedding.shape)

        for block in self.blocks:
            # print("ITERATIVE", embedding)
            embedding = block.forward(embedding, block_size, training=training)

        # print("AB", embedding)

        embedding = self.ln.forward(embedding)

        # print("BD", embedding)

        return self.head.forward(embedding)

    def backward(self, y):
        inner_d, loss = self.head.backward(y)

        inner_d = self.ln.backward(inner_d)

        for block in reversed(self.blocks):
            inner_d = block.backward(inner_d)

        self.token_embedding.backwards(inner_d)
        self.positional_embedding.backwards(inner_d)

        return loss

    def p_count(self):
        p_count = self.token_embedding.p_count() + self.positional_embedding.p_count()
        for block in self.blocks:
            p_count += block.p_count()
        return p_count + self.ln.p_count() + self.head.p_count()

    def grad_norm(self, its):
        gi = its * self.block_size

        grad_sq = self.token_embedding.grad_sq(gi) + self.positional_embedding.grad_sq(gi)
        for block in self.blocks:
            grad_sq += block.grad_sq(gi)
        grad_sq += self.ln.grad_sq(gi) + self.head.grad_sq(gi)

        return np.sqrt(grad_sq)

    def set_lr(self, lr):
        self.lr = lr

        self.token_embedding.set_lr(lr)
        self.positional_embedding.set_lr(lr)

        for block in self.blocks:
            block.set_lr(lr)

        self.ln.set_lr(lr)
        self.head.set_lr(lr)

    def update(self, its, t):
        grad_norm = self.grad_norm(its)
        if grad_norm > 1:
            its = its * grad_norm

        self.token_embedding.update(its * self.block_size, t)
        self.positional_embedding.update(its * self.block_size, t)

        for block in self.blocks:
            block.update(its * self.block_size, t)

        self.ln.update(its * self.block_size, t)
        self.head.update(its * self.block_size, t)

        # print("equal weights", np.all(self.token_embedding.weights == self.head.weights))

    def generate(self, ip, length):
        for i in range(length):
            x = self.forward(np.array([ip]), ip.shape[0], training=False)
            # print("X", x.shape, x)
            next_idx = np.argmax(np.random.multinomial(1, x[0][-1]))
            # print("x", x[-1], np.argmax(x[-1]))
            ip = np.append(ip, next_idx)
            ip = ip[-self.block_size:]
            # print("NIP", ip)

        return ip

import cupy as np


class Layer:
    def __init__(self, n_in, n_la, act, lr=0.01, beta_1=0.9, beta_2=0.999, eps=1e-5, weight_decay=1e-1):
        self.n_in = n_in
        self.n_la = n_la

        self.act = act

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weight_decay = weight_decay

        self.biases = np.zeros(n_la)
        self.weights = np.random.normal(size=(n_la, n_in), scale=np.sqrt(2 / (n_in + n_la)))

        self.input = []
        self.results = []

        self.bc = np.zeros_like(self.biases)
        self.wc = np.zeros_like(self.weights)

        self.bm = np.zeros_like(self.biases)
        self.wm = np.zeros_like(self.weights)

        self.bv = np.zeros_like(self.biases)
        self.wv = np.zeros_like(self.weights)

    def forward(self, x):
        self.input = x

        z = self.biases + np.matmul(x, self.weights.T)
        a = self.act.a(z)

        self.results = a
        return a

    def backward(self, inner_d):
        inner_d *= self.act.a_p(self.results)

        self.bc += np.sum(inner_d, axis=(0, 1))
        self.wc += np.sum(np.matmul(inner_d.transpose(0, 2, 1), self.input), axis=0)

        inner_d = np.matmul(inner_d, self.weights)

        return inner_d

    def p_count(self):
        return self.biases.size + self.weights.size

    def grad_sq(self, gi):
        return np.sum((self.bc / gi) ** 2) + np.sum((self.wc / gi) ** 2)

    def set_lr(self, lr):
        self.lr = lr

    def update(self, its, t=0):
        m_correct = 1 / (1 - self.beta_1 ** (t + 1))
        v_correct = 1 / (1 - self.beta_2 ** (t + 1))

        self.bc /= its
        self.wc /= its

        self.bm = self.beta_1 * self.bm + (1 - self.beta_1) * self.bc
        self.wm = self.beta_1 * self.wm + (1 - self.beta_1) * self.wc

        self.bv = self.beta_2 * self.bv + (1 - self.beta_2) * self.bc ** 2
        self.wv = self.beta_2 * self.wv + (1 - self.beta_2) * self.wc ** 2

        self.biases -= (self.bm * m_correct) * self.lr / (np.sqrt(self.bv * v_correct) + self.eps)
        self.weights -= (self.wm * m_correct) * self.lr / (np.sqrt(self.wv * v_correct) + self.eps)
        self.weights *= (1 - self.lr * self.weight_decay)

        self.reset()

    def reset(self):
        self.bc = np.zeros_like(self.biases)
        self.wc = np.zeros_like(self.weights)

import cupy as np


class Encode:
    def __init__(self, n_in, n_en, lr=0.01, beta_1=0.9, beta_2=0.999, eps=1e-5, weight_decay=1e-1):
        self.n_in = n_in
        self.n_la = n_en

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weight_decay = weight_decay

        self.weights = np.random.normal(size=(n_en, n_in), scale=np.sqrt(2 / (n_en + n_in)))

        self.input = []
        self.results = []

        self.wc = np.zeros_like(self.weights)

        self.wm = np.zeros_like(self.weights)

        self.wv = np.zeros_like(self.weights)

    def forward(self, x):
        self.input = x
        # print("INTO CH", x)
        # print("WG", self.weights)

        z = np.matmul(x, self.weights.T)
        # print("RES", z)
        self.results = z
        return z

    def backward(self, inner_d):
        self.wc += np.sum(np.matmul(inner_d.transpose(0, 2, 1), self.input), axis=0)

        inner_d = np.matmul(inner_d, self.weights)

        return inner_d

    def p_count(self):
        return self.weights.size

    def grad_sq(self, gi):
        return np.sum((self.wc / gi) ** 2)

    def set_lr(self, lr):
        self.lr = lr

    def update(self, its, t=0):
        m_correct = 1 / (1 - self.beta_1 ** (t + 1))
        v_correct = 1 / (1 - self.beta_2 ** (t + 1))

        self.wc /= its

        self.wm = self.beta_1 * self.wm + (1 - self.beta_1) * self.wc

        self.wv = self.beta_2 * self.wv + (1 - self.beta_2) * self.wc ** 2

        self.weights -= (self.wm * m_correct) * self.lr / (np.sqrt(self.wv * v_correct) + self.eps)
        self.weights *= (1 - self.lr * self.weight_decay)

        self.reset()

    def reset(self):
        self.wc = np.zeros_like(self.weights)

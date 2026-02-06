import cupy as np


class LayerNorm:
    def __init__(self, n_in, lr=0.01, beta_1=0.9, beta_2=0.999, eps=1e-5):
        self.n_in = n_in

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        self.beta = np.zeros(n_in)
        self.gamma = np.ones(n_in)

        self.bc = np.zeros_like(self.beta)
        self.gc = np.zeros_like(self.gamma)

        self.bm = np.zeros_like(self.beta)
        self.gm = np.zeros_like(self.gamma)

        self.bv = np.zeros_like(self.beta)
        self.gv = np.zeros_like(self.gamma)

        self.mean = []
        self.sigma = []

        self.input = []
        self.results = []

    def forward(self, x):
        self.input = x

        # print("I", x)
        self.mean = np.mean(x, axis=-1, keepdims=True)
        # print("mean", self.mean)
        # print("SWQ", (x - self.mean) ** 2)
        var = np.mean((x - self.mean) ** 2, axis=-1, keepdims=True)
        # print("SIG", self.sigma)
        self.sigma = np.sqrt(var + self.eps)

        normed = (x - self.mean) / self.sigma

        self.results = normed * self.gamma + self.beta
        # print("RES", self.results, self.results[0], np.mean(self.results[0]), np.var(self.results[0]))

        return self.results

    def backward(self, inner_d):
        normed = (self.input - self.mean) / self.sigma

        dx_hat = inner_d * self.gamma
        self.bc += np.sum(inner_d, axis=(0, 1))
        self.gc += np.sum(inner_d * normed, axis=(0, 1))

        dx = (dx_hat - np.mean(dx_hat, axis=-1, keepdims=True)
              - normed * np.mean(dx_hat * normed, axis=-1, keepdims=True)) / self.sigma

        return dx

    def p_count(self):
        return self.beta.size + self.gamma.size

    def grad_sq(self, gi):
        return np.sum((self.bc / gi) ** 2) + np.sum((self.gc / gi) ** 2)

    def set_lr(self, lr):
        self.lr = lr

    def update(self, its, t=0):
        m_correct = 1 / (1 - self.beta_1 ** (t + 1))
        v_correct = 1 / (1 - self.beta_2 ** (t + 1))

        self.bc /= its
        self.gc /= its

        self.bm = self.beta_1 * self.bm + (1 - self.beta_1) * self.bc
        self.gm = self.beta_1 * self.gm + (1 - self.beta_1) * self.gc

        self.bv = self.beta_2 * self.bv + (1 - self.beta_2) * self.bc ** 2
        self.gv = self.beta_2 * self.gv + (1 - self.beta_2) * self.gc ** 2

        self.beta -= (self.bm * m_correct) * self.lr / (np.sqrt(self.bv * v_correct) + self.eps)
        self.gamma -= (self.gm * m_correct) * self.lr / (np.sqrt(self.gv * v_correct) + self.eps)

        self.reset()

    def reset(self):
        self.bc = np.zeros_like(self.beta)
        self.gc = np.zeros_like(self.gamma)

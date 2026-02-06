import cupy as np
from util import Softmax, CrossEntropy


class TiedLayer:
    def __init__(self, n_in, n_la, tied_layer, lr=0.01, beta_1=0.9, beta_2=0.999, eps=1e-5, weight_decay=1e-1):
        self.n_in = n_in
        self.n_la = n_la

        self.act = Softmax

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weight_decay = weight_decay

        self.biases = np.zeros(n_la)
        self.tied_layer = tied_layer

        self.input = []
        self.results = []

        self.bc = np.zeros_like(self.biases)

        self.bm = np.zeros_like(self.biases)

        self.bv = np.zeros_like(self.biases)

    def forward(self, x):
        self.input = x

        # print("SX", x.shape, "SW", self.weights.shape, x.T.shape)
        # print("x", x)
        #  print("w", self.weights)
        # print("dot", np.matmul(x, self.weights.T))
        z = self.biases + np.matmul(x, self.tied_layer.weights.T)
        a = self.act.a(z)

        self.results = a
        return a

    def backward(self, targets):
        # print("res", self.results)
        # print("t", targets)
        loss = CrossEntropy.a(self.results, targets)
        # print("LOSS", loss)
        inner_d = CrossEntropy.a_p(self.results, targets)

        # print("id", inner_d)
        # print("s", np.sum(inner_d, axis=0))

        self.bc += np.sum(inner_d, axis=(0, 1))
        self.tied_layer.wc += np.sum(np.matmul(inner_d.transpose(0, 2, 1), self.input), axis=0)

        # flatten before
        # B, T, C_out = inner_d.shape
        # C_in = self.input.shape[-1]

        # flat_inner_d = inner_d.reshape(B * T, C_out)
        # inp = self.input.reshape(B * T, C_in)

        # self.bc += np.sum(flat_inner_d, axis=0)
        # self.wc += np.matmul(flat_inner_d.T, inp)

        # print(self.weights.T.shape, inner_d.shape, inner_d)
        inner_d = np.matmul(inner_d, self.tied_layer.weights)
        # print("aid", inner_d)

        return inner_d, loss

    def p_count(self):
        return self.biases.size

    def grad_sq(self, gi):
        return np.sum((self.bc / gi) ** 2)

    def set_lr(self, lr):
        self.lr = lr

    def update(self, its, t=0):
        m_correct = 1 / (1 - self.beta_1 ** (t + 1))
        v_correct = 1 / (1 - self.beta_2 ** (t + 1))

        self.bc /= its

        self.bm = self.beta_1 * self.bm + (1 - self.beta_1) * self.bc

        self.bv = self.beta_2 * self.bv + (1 - self.beta_2) * self.bc ** 2

        self.biases -= (self.bm * m_correct) * self.lr / (np.sqrt(self.bv * v_correct) + self.eps)

        self.reset()

    def reset(self):
        self.bc = np.zeros_like(self.biases)

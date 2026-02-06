import cupy as np


class N:
    @staticmethod
    def a(z):
        return z

    @staticmethod
    def a_p(a):
        return np.ones_like(a)


class Sigmoid:
    @staticmethod
    def a(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def a_p(a):
        return a * (1 - a)


class ReLU:
    @staticmethod
    def a(z):
        return np.maximum(z, 0)

    @staticmethod
    def a_p(a):
        return (a > 0).astype(a.dtype)


class Softmax:
    @staticmethod
    def a(z):
        # print("SI", z)
        exp = np.exp(z - np.max(z, axis=-1, keepdims=True))
        # print("post", exp)
        # print("TRUE", np.max(z), np.max(z, axis=-1), np.max(z, axis=-1, keepdims=True))
        # print("SUM", np.sum(exp, axis=-1), np.sum(exp, axis=-1, keepdims=True))
        # print("N", exp / np.sum(exp, axis=-1, keepdims=True))
        # print("OLD", np.exp(z - np.max(z)) / np.sum(exp))
        return exp / np.sum(exp, axis=-1, keepdims=True)

    @staticmethod
    def a_p(a, g):
        # print("A", a)
        # print("G", g)
        s = np.sum(g * a, axis=-1, keepdims=True)
        # print("S", s)
        # print("COMP", g * a)
        inner_d = a * (g - s)
        # print("ID", inner_d)

        return inner_d


class CrossEntropy:
    @staticmethod
    def a(z, y):
        B, T, C = z.shape
        # print("ZMX", z)
        # print("NOT ADJ", np.log(np.sum(np.exp(z))), np.sum(np.exp(z)), np.exp(z), np.sum(np.exp(z), axis=-1))
        # print("fin", np.log(np.sum(np.exp(z), axis=-1)))
        # print("YS", -z[y], "Z", z, "Y", y)
        # for cross entropy with a, not z
        # return -a[np.arange(len(y)), y] + np.log(np.sum(np.exp(a), axis=-1))
        # print("ZS", z.shape)
        # print("YS", y.shape)

        return - np.log(z[np.arange(B)[:, None], np.arange(T)[None, :], y])

    @staticmethod
    def a_p(y_hat, y):
        B, T, C = y_hat.shape
        # print("YS", y_hat.shape)
        # print("BG", y_hat)
        # does not copy but modifies
        inner_d = y_hat.copy()
        inner_d[np.arange(B)[:, None], np.arange(T)[None, :], y] -= 1
        # print("AF", inner_d)
        return inner_d


class MSE:
    @staticmethod
    def c(y_hat, y):
        return (y_hat - y) ** 2 / y_hat.size

    @staticmethod
    def c_p(y_hat, y):
        return 2 * (y_hat - y) / y_hat.size

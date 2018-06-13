# python: 2.7
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pickle


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sample_sigmoid(prob):
    return np.random.binomial(1, prob)


class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden=2, n_observe=784):
        """Initialize model."""
        self.nh = n_hidden
        self.nv = n_observe
        self.W = np.zeros((n_observe, n_hidden))
        self.a = np.zeros((n_observe, 1))
        self.b = np.zeros((n_hidden, 1))

    def train(self, data, T=10, learning_rate=0.01, batch_size=16):
        """Train model using data."""
        # Reshape the training data
        data = np.reshape(data, (-1, self.nv))

        # Number of training e.g.
        N = data.shape[0]
        # learning_rate
        alpha = learning_rate

        # Variables of the model
        W, a, b = self.W, self.a, self.b
        # The hidden variables(vec)
        h = np.zeros((self.nh, batch_size))
        v = np.zeros((self.nv, batch_size))
        hp = np.zeros((self.nh, batch_size))  # h_prime
        vp = np.zeros((self.nv, batch_size))  # v_prime

        for t in range(T):
            num_batch = int(N / batch_size)
            for n in range(num_batch):
                if (n + 1) * batch_size <= N:
                    end = (n + 1) * batch_size
                else:
                    end = N
                v = data[n * batch_size : end, :].T

                p_hv = sigmoid(np.dot(W.T, v) + b)
                h = sample_sigmoid(p_hv)

                pos_grad = np.dot(v, h.T)

                p_vh = sigmoid(np.dot(W, h) + a)
                vp = sample_sigmoid(p_vh)
                
                p_hv = sigmoid(np.dot(W.T, vp) + b)
                hp = sample_sigmoid(p_hv)

                neg_grad = np.dot(vp, hp.T)

                # Update params: W, a, b
                W += alpha/batch_size * (pos_grad - neg_grad)
                # print(((v - vp).shape))
                # avg = np.average(v - vp, axis=1)
                # print(avg.shape)
                a += alpha / batch_size * np.sum(v - vp, axis=1, keepdims=True)
                b += alpha / batch_size * np.sum(h - hp, axis=1, keepdims=True)

        self.W, self.a, self.b = W, a, b

    def sample(self, n=1, T=10):
        """Sample from trained model."""
        W, a, b = self.W, self.a, self.b

        v = np.random.randn(self.nv, n)
        p_hv = sigmoid(np.dot(W.T, v) + b)
        h = sample_sigmoid(p_hv)

        for t in range(T):
            p_vh = sigmoid(np.dot(W, h) + a)
            v = sample_sigmoid(p_vh)
            p_hv = sigmoid(np.dot(W.T, v) + b)
            h = sample_sigmoid(p_hv)

        v = v.T
        return v.reshape((n, 28, 28))


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_model(filename):
    """
    Return model
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


# train restricted boltzmann machine using mnist dataset
if __name__ == '__main__':
    # load mnist dataset, no label
    mnist = np.load('mnist_bin.npy')  # 60000x28x28
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols
    print(mnist.shape)

    # # construct rbm model
    # rbm = RBM(10, img_size)

    # # train rbm model using mnist
    # rbm.train(mnist)

    # # sample from rbm model
    # s = rbm.sample()

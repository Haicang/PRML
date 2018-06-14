# python: 2.7
# encoding: utf-8

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sample_sigmoid(prob):
    return np.random.binomial(1, prob)


def sample_sigmoid_torch(prob):
    return torch.bernoulli(prob)


class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden=2, n_observe=784):
        """Initialize model."""
        self.nh = n_hidden
        self.nv = n_observe
        self.W = np.zeros((n_observe, n_hidden))
        self.a = np.zeros((n_observe, 1))
        self.b = np.zeros((n_hidden, 1))

    def train(self, data, T=10, learning_rate=0.01, batch_size=16, log=False):
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
            # To shuffle in a epoch
            indexes = np.random.permutation(num_batch)
            # To indicate the percentage of an epoch
            batch = 0
            for n in indexes:
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

                batch += 1
                if log is True and n % 1000 == 0:
                    print('Epoch %2d/%2d -- %d/%d' %(t+1, T, n, num_batch))

        self.W, self.a, self.b = W, a, b

    def sample(self, n=1, T=10, spciman=None):
        """
        Sample from trained model.
        n: the number of samples, only used when spciman is None
        spciman: some data from MNIST, of shape (n, 28, 28)
        """
        W, a, b = self.W, self.a, self.b

        if spciman is None:
            v = np.random.randn(self.nv, n)
        else:
            v = spciman.reshape(-1, 784).T
        p_hv = sigmoid(np.dot(W.T, v) + b)
        h = sample_sigmoid(p_hv)

        for t in range(T):
            p_vh = sigmoid(np.dot(W, h) + a)
            v = sample_sigmoid(p_vh)
            p_hv = sigmoid(np.dot(W.T, v) + b)
            h = sample_sigmoid(p_hv)

        v = v.T
        v[v > 0.5] = 1; v[v <= 0.5] = 0
        return v.reshape((n, 28, 28))


class RBMtorch():
    """
    Restricted Boltzmann Machine using pytorch
    """
    def __init__(self, n_hidden=2, n_observe=784):
        self.nv = n_observe
        self.nh = n_hidden
        self.W = torch.zeros(self.nv, self.nh, dtype=torch.double)
        self.a = torch.zeros(n_observe, 1, dtype=torch.double)
        self.b = torch.zeros(n_hidden, 1, dtype=torch.double)

    def train(self, data, T=10, learning_rate=0.005, batch_size=16, log=False, gpu=False):
        """Train model using data."""
        if gpu is True and torch.cuda.is_available():
            gpu = True

        if gpu is False:
            # Reshape the training data
            data = torch.DoubleTensor(np.reshape(data, (-1, self.nv)))

            # Number of training e.g.
            N = data.shape[0]
            # learning_rate
            alpha = learning_rate

            # Variables of the model
            W, a, b = self.W, self.a, self.b
            # The hidden variables(vec)
            h = torch.zeros(self.nh, batch_size, dtype=torch.double)
            v = torch.zeros(self.nv, batch_size, dtype=torch.double)
            hp = torch.zeros(self.nh, batch_size, dtype=torch.double)  # h_prime
            vp = torch.zeros(self.nv, batch_size, dtype=torch.double)  # v_prime
        else:
            # Reshape the training data
            data = torch.cuda.DoubleTensor(np.reshape(data, (-1, self.nv)))

            # Number of training e.g.
            N = data.shape[0]
            # learning_rate
            alpha = learning_rate

            # Variables of the model
            W, a, b = self.W.cuda(), self.a.cuda(), self.b.cuda()
            # The hidden variables(vec)
            h = torch.zeros(self.nh, batch_size, dtype=torch.double).cuda()
            v = torch.zeros(self.nv, batch_size, dtype=torch.double).cuda()
            hp = torch.zeros(self.nh, batch_size,
                             dtype=torch.double).cuda()  # h_prime
            vp = torch.zeros(self.nv, batch_size,
                             dtype=torch.double).cuda()  # v_prime

        for t in range(T):
            num_batch = int(N / batch_size)
            # To shuffle in a epoch, 
            indexes = torch.randperm(num_batch).cuda()
            # To indicate the percentage of an epoch
            batch = 0
            for n in indexes:
                if (n + 1) * batch_size <= N:
                    end = (n + 1) * batch_size
                else:
                    end = N
                v = data[n * batch_size: end, :].t()

                p_hv = torch.sigmoid(torch.mm(W.t(), v) + b)
                h = torch.bernoulli(p_hv)

                pos_grad = torch.mm(v, h.t())

                p_vh = torch.sigmoid(torch.mm(W, h) + a)
                vp = torch.bernoulli(p_vh)


                p_hv = torch.sigmoid(torch.mm(W.t(), vp) + b)
                hp = torch.bernoulli(p_hv)

                neg_grad = torch.mm(vp, hp.t())

                # Update params: W, a, b
                W += alpha/batch_size * (pos_grad - neg_grad)
                # print(((v - vp).shape))
                # avg = np.average(v - vp, axis=1)
                # print(avg.shape)
                a += alpha / batch_size * torch.sum(v - vp, 1, keepdim=True)
                b += alpha / batch_size * torch.sum(h - hp, 1, keepdim=True)

                batch += 1
                if log is True and batch % 1000 == 0:
                    print('Epoch [%d/%d] -- %d/%d' % (t+1, T, batch, num_batch))
            # if log is True:
            #     print('Epoch: %d/%d' % (t+1, T))

        self.W, self.a, self.b = W.cpu(), a.cpu(), b.cpu()

    def sample(self, n=1, T=10, spciman=None):
        """
        Sample from trained model.
        n: the number of samples, only used when spciman is None
        spciman: some data from MNIST, of shape (n, 28, 28)
        """
        W, a, b = self.W, self.a, self.b

        if spciman is None:
            v = torch.randn(self.nv, n, dtype=torch.double)
        else:
            v = torch.DoubleTensor(spciman.reshape(-1, 784)).t()
        p_hv = torch.sigmoid(torch.mm(W.t(), v) + b)
        h = torch.bernoulli(p_hv)

        for t in range(T):
            p_vh = torch.sigmoid(torch.mm(W, h) + a)
            v = torch.bernoulli(p_vh)
            p_hv = torch.sigmoid(torch.mm(W.t(), v) + b)
            h = torch.bernoulli(p_hv)

        v = v.t()
        v[v > 0.5] = 1
        v[v <= 0.5] = 0
        return v.reshape(-1, 28, 28)


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

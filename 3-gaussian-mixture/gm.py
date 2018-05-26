

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

import copy


def generate_data():
    """
    Generate data for the Gaussian Mixture Model
    """
    means = ([5, 5], [10, 10], [15, 15])
    covs = ([[1, 0.1], [0.2, 2]],  [[3, 0], [0, 3]], [[1, 0], [0.5, 1]])

    k = len(means)
    size = 200

    points = []
    for (mean, cov) in zip(means, covs):
        scats = np.random.multivariate_normal(mean, cov, size)
        points.append(scats)
    data = np.concatenate(points)

    # random shuffle the dataset
    perm = np.random.permutation(range(data.shape[0]))
    data = data[perm, :]

    return data


# Utils functions for K-means

def distortion(X, label, mu):
    """
    To compute \sum ||x_i - mu_{c^i}||^2 in k-means
    """
    S = 0
    m = X.shape[0]
    for i in range(m):
        d = X[i, :] - mu[label[i], :]
        S += np.inner(d, d)
    return S


def tag(X, mu):
    """
    To tag points in k-means
    """
    m = X.shape[0]
    k = mu.shape[0]
    c = np.zeros((m, ))
    for i in range(m):
        d = X[i] - mu[0]
        dist = np.inner(d, d)
        c[i] = 0
        for j in range(1, k):
            d = X[i] - mu[j]
            tmp = np.inner(d, d)
            if tmp < dist:
                dist = tmp
                c[i] = j

    return c.astype(np.int32)


class KMeans():

    def __init__(self, n_cluster, max_iter=300, tol=0.0001):
        self.n = n_cluster
        self.epochs = max_iter
        self.tol = tol

        self.mu = None

    def fit(self, X):

        def update_mu(X, c, mu):
            """
            To update mu in K-means
            """
            k = mu.shape[1]
            for j in range(k):
                indices = (c == j)
                count = np.sum(indices)
                mu[j, :] = np.sum(X[indices, :], axis=0) / count
            return mu

        # assert X.shape
        k = self.n
        size = X.shape[0]
        # Random init
        step = int(size / k)
        mu = [np.mean(X[i: i + step, :], axis=0) for i in range(0, size, step)]
        mu = np.array(mu)

        assert mu.shape[1] == X.shape[1]

        c = tag(X, mu)
        J = distortion(X, c, mu)

        for e in range(self.epochs):
            c = tag(X, mu)
            mu = update_mu(X, c, mu)

            J_new = distortion(X, c, mu)
            if np.abs(J - J_new) < self.tol:
                break
            J = J_new

        self.mu = mu

    def predict(self, X):
        return tag(X, self.mu)


def Estep(X, phi, mu, sigma):
    m = X.shape[0]
    k = mu.shape[0]
    P = np.zeros((m, k))

    for j in range(k):
        m_norm = multivariate_normal(mean=mu[j], cov=sigma[j])
        P[:, j] = m_norm.pdf(X) * phi[j]
    W = P / np.sum(P, axis=1).reshape(m, 1)

    return W


def Mstep(X, W, phi, mu, sigma):
    # update phi
    m = W.shape[0]
    phi = np.sum(W, axis=0) / m

    # update mu
    (m, k) = W.shape
    for j in range(k):
        #         print(W.shape, X.shape)
        col = np.reshape(W[:, j], (-1, 1))
        mu[j] = np.sum(col * X, axis=0) / np.sum(col, axis=0)
#         Sum = np.sum(W[:, j] * X, axis=0)
#         mu[j] = Sum / np.sum(W[:, j], axis=0)

    # update sigma
    for j in range(k):
        for i in range(m):
            d = X[i] - mu[j]
            sigma[j] += W[i, j] * np.outer(d, d)
        sigma[j] /= np.sum(W[:, j])

    return phi, mu, sigma


class GaussianMixture():

    def __init__(self, n_cluster, max_iter=100, tol=0.0001):
        self.n = n_cluster
        self.epochs = max_iter
        self.tol = tol

        # params
        self.phi = None
        self.mu = None
        self.sigma = None

    def fit(self, X, init='kmeans', init_iter=10):
        """
        Init the fit with `kmeans`, 
        but can be set as None, and use random init
        """

        # These functions are replaced by `Estep` and `Mstep`
        def update_W():
            pass

        def update_phi(W):
            m = W.shape[0]
            return np.sum(W, axis=0) / m

        def update_mu(W, X, mu):
            (m, k) = W.shape
            for j in range(k):
                mu[j] = np.sum(W[:, j] * X, axis=0) / np.sum(W[:, j], axis=0)
            return mu

        def update_sigma(W, X, mu, sigma):
            for j in range(k):
                for i in range(m):
                    d = X[i] - mu[j]
                    sigma[j] += W[i, j] * np.outer(d, d)
                sigma[j] /= np.sum(W[:, j])
            return sigma

        m = X.shape[0]
        n = X.shape[1]
        k = self.n

        # Init params with k-means
        if init == 'kmeans':
            max_iter = init_iter
        else:
            max_iter = 0

        kmeans = KMeans(k, max_iter=0)
        kmeans.fit(X)
        mu = copy.deepcopy(kmeans.mu)

        W = np.zeros((m, k))
        c = kmeans.predict(X)
        for i in range(m):
            W[i, c[i]] = 1

        phi = update_phi(W)

        sigma = np.zeros((k, n, n))
        sigma = update_sigma(W, X, mu, sigma)

        # Use EM alg to iterate
        for e in range(self.epochs):
            W = Estep(X, phi, mu, sigma)
            phi, mu, sigma = Mstep(X, W, phi, mu, sigma)

        # Save params to class
        self.phi = phi
        self.mu = mu
        self.sigma = sigma

    def predict(self, X):
        W = Estep(X, self.phi, self.mu, self.sigma)
        return np.argmax(W, axis=1)


if __name__ == '__main__':
    pass

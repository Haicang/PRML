

import numpy as np
import matplotlib.pyplot as plt


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
    sum = 0
    m = X.shape[0]
    for i in range(m):
        d = X[i, :] - mu[label[i], :]
        sum += np.inner(d, d)
    return sum


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

    return np.int32(c)


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


class KMeans():

    def __init__(self, n_cluster, max_iter=300, tol=0.0001):
        self.n = n_cluster
        self.epochs = max_iter
        self.tol = tol

        self.mu = None

    def fit(self, X):
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


class GaussianMixture():
    pass


if __name__ == '__main__':
    pass

# python: 3.5.2
# encoding: utf-8

import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp


# define some kernel functions here
# All the return matrix will be of shape(#x1, #x2)
def gaussian(x1, x2, sigma=1):
    """
    Gaussian kernel
    x1 and x2 should be np.ndarray of shape (n, ) or (n, x)
    """
    n, m = x1.shape[0], x2.shape[0]
    K = np.zeros((n, m))

    for j in range(m):
        # Compute a column each time
        x = np.linalg.norm(x1 - x2[j,:], axis=1, keepdims=False)
        K[:, j] = np.exp(- x ** 2 / (2 * sigma ** 2))

    return K


def linear(x1, x2):
    """
    Linear kernel
    """
    return np.dot(x1, x2.T)


def poly(x1, x2, p=2):
    """
    Poly kernel
    """
    lin = 1 + np.dot(x1, x2.T)
    return lin ** p


Kernels = {'g': gaussian, 'l': linear, 'p': poly}
# Kernel functions define ends


def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算准确率。
    """
    return np.sum(label == pred) / len(pred)


class SVM():
    """
    SVM模型。
    """

    def __init__(self, C=1, n_classes=2, kernel='g', loss=None):
        """
        C: the hyperparameter in svm for overlap distribution
        n_classes: the number of classification class
        kernel: {'g': gaussian, 'l': linear, 'p': ploy}
        loss: None for Lagrange max, 'hinge' for hinge loss(only for linear)
        """
        self.n = n_classes
        self.kernel = Kernels[kernel]
        self.C = C

        # a.shape = (m, 1)
        # only support vectors' a are stored,
        self.num = None  # real number
        self.index = None
        self.a = None
        self.b = None  # real number
        self.t = None
        self.x = None

        self.loss = loss

    def train(self, data_train):
        """
        训练模型。
        """
        x_train = data_train[:, :2]  # feature [x1, x2]
        t_train = data_train[:, 2]  # 真实标签

        if self.loss == None:
            self.fit_kernel(x_train, t_train)
        elif self.loss == 'hinge':
            # Only linear kernel is valid for hinge loss
            assert self.kernel == linear
            self.fit_hinge(x_train, t_train)
        else:
            raise Exception

        pass

    def predict(self, x):
        """
        预测标签。
        """
        # n = x.shape[0]
        K = self.kernel(self.x, x)  # (num, n)
        a = self.a.reshape(-1, 1)
        t = self.t.reshape(-1, 1)
        y = np.sum(a * t * K) + self.b
        return y.T

    def fit_kernel(self, X, t):
        """
        Use the Lagrange
        """
        n = X.shape[0]
        t = t.reshape(n, 1)

        # Get the Kernel matrix for the whole training set (Gram matrix)
        K = self.kernel(X, X)

        # maximize the L is minimize -L
        P = matrix(np.dot(t, t.T) * K)
        q = matrix(- np.ones(shape=(n, 1)))

        I = np.identity(n)
        G = np.zeros((2 * n, n))
        G[0:n, :] = I
        G[n: , :] = -I
        G = matrix(G)
        h = np.zeros((2 * n, 1))
        h[0:n, :] = self.C
        h = matrix(h)

        A = matrix(np.diag(t.squeeze()))
        b = matrix(np.zeros((n, 1)))

        sol = qp(P, q, G, h, A, b)
        a = np.array(sol['x'])
        a = a.squeeze()
        print(sol)
        return sol

        self.a = a[a != 0]
        index = np.array(range(n))
        self.index = index[a != 0]
        self.t = t[self.index, 0]
        self.num = self.index.shape[0]
        print(self.num)

        # use the support vectors to calcuate intercept, 's' means 'support'
        self.x = X[self.index, :]
        K_s = self.kernel(self.x, self.x)
        col = self.a.reshape(1, -1) * t[self.x, :].reshape(1, -1) * K_s
        col = np.sum(col, axis=1, keepdims=False)
        self.b = 1./self.num * np.sum(np.squeeze(t) - col, keepdims=False)


    def fit_hinge(self, X, t):
        """
        Use the hinge loss for linear kernel
        """
        pass


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    svm = SVM()  # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))


# quit the matlab engine
# eng.quit()

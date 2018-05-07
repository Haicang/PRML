# python: 3.5.2
# encoding: utf-8

import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp
import matplotlib.pyplot as plt
from utils import *


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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
        n_classes: the number of classification class, this class only support bi-classes
        kernel: {'g': gaussian, 'l': linear, 'p': ploy}
        loss: None for Lagrange max, 'hinge' for hinge loss(only for linear)
        decision_function_shape
        """
        self.n = n_classes
        self.kernel = Kernels[kernel]
        self.C = C

        # a.shape = (m, 1)
        # only support vectors' a are stored,
        # number of support vectors
        self.num = None  # real number
        # support vectors' indexes
        self.index = None
        # support vectors' coef
        self.a = None
        # intercept
        self.b = None  # real number
        # support vectors' targets
        self.t = None
        # support vectors
        self.x = None

        self.w = None  # only used for linear kernel in hinge loss

        self.loss = loss

    def train(self, data_train, epochs=1000):
        """
        训练模型。
        """
        n = data_train.shape[1] - 1
        x_train = data_train[:, :n]  # feature [x1, x2]
        t_train = data_train[:, n]  # 真实标签

        self.fit(x_train, t_train, epochs)

    def fit(self, X, t, epochs=1000):
        """
        Train the model
        X: (n, 2)
        t: (n, )
        """
        if self.loss == None:
            self.fit_kernel(X, t)
        elif self.loss == 'hinge':
            # Only linear kernel is valid for hinge loss
            # self.kernel = linear
            assert self.kernel == linear
            self.fit_hinge(X, t, epochs)
        else:
            raise Exception('Loss function not found.')

        pass

    def predict(self, x):
        """
        预测标签。
        x: (m, 2)
        """
        # n = x.shape[0]
        if self.loss == None:
            K = self.kernel(self.x, x)  # (num, n)
            a = self.a.reshape(-1, 1)
            t = self.t.reshape(-1, 1)
            y = np.sum(a * t * K, axis=0) + self.b
            y = y.squeeze()
        elif self.loss == 'hinge':
            X = x.T  # (2, m)
            Y = np.dot(self.w.reshape(1, -1), X) + self.b  # (1, m)
            y = Y.squeeze()  # (m, 1)
        else:
            raise Exception

        y[y >= 0] = 1
        y[y < 0] = -1
        return y

    def fit_kernel(self, X, t):
        """
        Use the Lagrange
        """
        n = X.shape[0]
        t = t.reshape(n, 1)

        # Get the Kernel matrix for the whole training set (Gram matrix)
        K = self.kernel(X, X)
        # print(K)

        # maximize the L is minimize -L
        # 1/2 x^T P x + q^T x
        P = matrix(np.dot(t, t.T) * K)
        q = matrix(- np.ones(shape=(n, 1)))

        # use for the bound of x
        # 0 <= x <= C
        I = np.identity(n)
        G = np.zeros((2 * n, n))
        G[:n, :] = I
        G[n:, :] = -I
        G = matrix(G)
        h = np.zeros((2 * n, 1))
        h[0:n, :] = self.C
        h = matrix(h)

        # \sum a_n = 0
        A = matrix(t.reshape(1, -1))
        b = matrix(0.0)

        sol = qp(P, q, G, h, A, b)
        a = np.array(sol['x'])
        a = a.squeeze()
        # print(sol)
        if sol['status'] is not 'optimal':
            raise NotOptimal('Quadratic Optimazation Fail')
        # print(a)
        
        # If the coef of support vectors are too small,
        # change them to 0.
        a[np.abs(a) < DELTA] = 0

        # Store support vectors coef
        self.a = a[a != 0]
        # support vectors' indexes
        index = np.array(range(n))
        self.index = index[a != 0]
        # support vectors' targets
        self.t = t[self.index, 0]
        # number of support vectors
        self.num = self.index.shape[0]
        # print(self.num)

        # use the support vectors to calcuate intercept, 's' means 'support'
        self.x = X[self.index, :]
        K_s = self.kernel(self.x, self.x)
        col = self.a.reshape(1, -1) * t[self.index, :].reshape(1, -1) * K_s
        col = np.sum(col, axis=1, keepdims=False)
        self.b = 1./self.num * np.sum(np.squeeze(self.t) - col, keepdims=False)

    def fit_hinge(self, X, t, epochs=1000, lr=0.0001, l = 0.01):
        """
        Use the hinge loss for linear kernel
        X: (m, 2)
        t: (m, )
        lr: learning_rate
        l: l-2 penalty
        m is # training e.g.
        Use gradient descent to optimize
        """
        # X:(2, m)  t:(1, m)
        X = X.T
        t = t.reshape(1, -1)
        m = X.shape[1]
        n = X.shape[0]

        # intialization
        w = np.random.randn(n, 1)
        b = 0

        for _ in range(epochs):
            # forward
            Y = np.dot(w.T, X) + b  # (1, n)
            Z = 1 - Y * t  # (1, n)

            # back
            dw = -X
            dw[:, Z.squeeze() <= 0] = 0
            dw = np.sum(dw, axis=1, keepdims=True)  # (n, 1)
            db = - np.ones((1, m))
            db[:, Z.squeeze() <= 0] = 0
            db = np.sum(db, keepdims=False)

            # update
            # use mean here
            w = (1 - 2 * l * lr / m) * w - lr * dw / m
            b = b - lr * db / m
        
        self.w = w.squeeze()
        self.b = b


class multiSVM():
    def __init__(self, C=1, n_classes=2, kernel='g', loss=None, 
        decision_function_shape='ovr'):
        self.C = C
        self.n_classes = n_classes
        self.kernel = kernel
        self.loss = loss
        self.dfs = decision_function_shape

        # To store multiple SVMs
        self.models = []
        # To store the labels for training data
        # will also be used in prediction
        
        # Either one-over-rest or one-over-one
        assert self.dfs in ('ovr', 'ovo')

        # For the case of `ovr`
        # models is a list<SVM> of len(list)=n_classes
        # i in range(0, n_classes)
        # this i-th class verse the rest SVM is models[i]
        if self.dfs == 'ovr':
            for _ in range(self.n_classes):
                self.models.append(
                    SVM(C, n_classes, kernel, loss))

        # For the case of 'ovo':
        # models is a list<SVM> of [[n-1], [n-2], ... , [1]], a triangle matrix
        # i, j both in range(0, n_classes)
        # assume i < j, the SVM between i-class and j-class (also j to i)
        # model[i][j - i - 1]
        elif self.dfs == 'ovo':
            for i in range(n_classes - 1):
                lst = []
                for j in range(i + 1, n_classes):
                    lst.append(
                        SVM(C, n_classes, kernel, loss))
                self.models.append(lst)

        # Because of assertion, this part cannot be reached
        else:
            raise Exception('Decision function shape Error!\n' + 
            'Use `ovr` or `ovo`')
    
    def train(self, data, epochs=100):
        """
        Train multiple models
        epochs is only used in hinge loss
        """
        # (m, n): m -- number training examples; n -- number of features
        # m = data.shape[0]
        n = data.shape[1] - 1
        # training set
        X_raw = data[:, :n]
        # training targets
        T_raw = data[:, n]
        self.fit(X_raw, T_raw, epochs)

    def fit(self, X, t, epochs=100):
        """
        Train multiple models
        epochs is only used in hinge loss
        """
        X_raw, T_raw = X, t
        # number of training e.g.
        m = X_raw.shape[0]
        
        # labels set
        self.labels = tuple(set(list(T_raw)))

        if self.dfs == 'ovr':
            T = np.zeros(m)
            for (i, label) in enumerate(self.labels):
                T[T_raw == label] = 1
                T[T_raw != label] = -1
                self.models[i].fit(X_raw, T, epochs)

        elif self.dfs == 'ovo':
            for i in range(self.n_classes - 1):
                a = self.labels[i]
                for j in range(i + 1, self.n_classes):
                    b = self.labels[j]

                    # Constructing the training set for very ovo SVM
                    X1 = X_raw[T_raw == a]
                    T1 = np.ones((X1.shape[0], ))
                    X2 = X_raw[T_raw == b]
                    T2 = -np.ones((X2.shape[0], ))
                    X = np.concatenate((X1, X2), axis=0)
                    T = np.concatenate((T1, T2), axis=0)

                    self.models[i][j - i - 1].fit(X, T, epochs)

        else:
            # Because of __init__, this part cannot be reached
            pass

    def predict(self, X):
        """
        X: (m, 2)
        """
        # numbers of test e.g.
        m = X.shape[0]

        # raw predictions based on many SVMs
        preds = np.zeros((m, self.n_classes))
        # choice the first positive label
        pred = np.zeros((m, ))

        if self.dfs == 'ovr':
            for (i, model) in enumerate(self.models):
                preds[:, i] = model.predict(X)
            # choice the last positive
            for i in range(self.n_classes):
                pred[preds[:, i] == 1] = self.labels[i]

        elif self.dfs == 'ovo':
            tmp_pred = np.zeros((m, ))
            for i in range(self.n_classes - 1):
                for j in range(i + 1, self.n_classes):
                    model = self.models[i][j - i - 1]
                    tmp_pred = model.predict(X)
                    preds[tmp_pred > 0, i] += 1
                    preds[tmp_pred < 0, j] += 1
            tmp_pred = np.argmax(preds, axis=1)
            # convert the index into the actual labels
            for (i, label) in enumerate(self.labels):
                pred[tmp_pred == i] = label
        else:
            # Because of __init__, this part cannot be reached
            pass

        return pred


class Linear():
    """
    Linear classifier with squared error
    
    loss = sum (y_n - t_n)^2  - lambda * w.T w
    
    Also use mean for it, that is actually use the mean square error (MSE)
    """
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y, epochs=100, lr=0.01, l=0.01, show_loss=False):
        """
        Train the linear classifier with the training set
        X: (m, n)  training features
        y: (m, )  training targets
        epochs: iterating times
        lr: learning rate
        l:  l-2 penalty
        """
        # reshape the training data
        # X:(n, m)  Y:(1, m)
        X = X.transpose()
        Y = y.reshape(1, -1)
        
        # m: number of training e.g.
        # n: number of features
        n, m = X.shape

        # parameters
        w = np.random.randn(n, 1)
        b = 0

        # losses
        if show_loss is True:
            losses = []

        for i in range(epochs):
            # forward
            # the predictions (1, m)
            P = np.dot(w.T, X) + b
            Z = P - Y

            # compute loss
            if show_loss is True:
                loss = np.squeeze(np.sum(Z ** 2, keepdims=False) + l * np.dot(w.T, w))
                losses.append(loss)

            # back
            dw = 2 * Z * X
            dw = np.sum(dw, axis=1, keepdims=True)
            db = np.sum(Z, axis=1, keepdims=False)

            # update
            w = (1 - 2 * l * lr / m) * w - lr * dw / m
            b = b - lr * db / m

            if show_loss is True and i % 100 == 0:
                print('{}, loss: {}'.format(i, loss))

        if show_loss is True:
            print(losses[len(losses) - 1])
            plt.plot(losses)

        self.w = w.squeeze()
        self.b = b

    def train(self, data, epochs=100, lr=0.01, l=0.01, show_loss=False):
        # n: the number of features
        n = data.shape[1] - 1
        X, y = data[:, 0:n], data[:, n]
        self.fit(X, y, epochs, lr, l, show_loss)

    def predict(self, X):
        """
        Give predictions on the trained model
        X: (m, n)  m(# training e.g.)  n(# features)

        return y: (m, )
        """
        # reshape
        X = X.transpose()  # (n, m)
        w = self.w.reshape(-1, 1)  # (n, 1)
        b = self.b

        y = np.dot(w.T, X) + b
        y = y.squeeze()
        y[y <  0] = -1
        y[y >= 0] = 1
        return y


class Logistic():
    """
    Logistic regression with cross entropy loss
    """
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y, epochs=1000, lr=0.01, l=0.01, show_loss=False):
        """
        Train the linear classifier with the training set
        X: (m, n)  training features
        y: (m, )  training targets
        epochs: iterating times
        lr: learning rate
        l:  l-2 penalty
        """
        # reshape the training data
        # X:(n, m)  Y:(1, m)
        X = X.transpose()
        Y = y.reshape(1, -1)

        # m: number of training e.g.
        # n: number of features
        n, m = X.shape

        # parameters
        w = np.random.randn(n, 1)
        b = 0

        # losses
        if show_loss is True:
            losses = []

        for i in range(epochs):
            # forward
            Z = np.dot(w.T, X) + b
            A = sigmoid(Z * Y)

            # compute loss
            if show_loss is True:
                loss = np.squeeze(
                    np.sum(Z ** 2, keepdims=False) + l * np.dot(w.T, w))
                losses.append(loss)

            # backprop
            # dA = - 1/A
            # dZ = dA * A * (1 - A) * Y
            dZ = -(1 - A) * Y
            dw = np.sum(dZ * X, axis=1, keepdims=True)
            db = np.sum(dZ, axis=1, keepdims=False)

            # update, use mean-cross entropy loss
            w = (1 - lr * 2 * l / m) * w - lr * dw / m
            b = b - lr * db / m

            if show_loss is True and i % 100 == 0:
                print('{}, loss: {}'.format(i, loss))

        if show_loss is True:
            print(losses[len(losses) - 1])
            plt.plot(losses)

        self.w = w.squeeze()
        self.b = b

    def train(self, data, epochs=1000, lr=0.01, l=0.01, show_loss=False):
        # n: the number of training features
        n = data.shape[1] - 1
        X, y = data[:, 0:n], data[:, n]
        self.fit(X, y, epochs, lr, l, show_loss)

    def predict(self, X):
        """
        Give predictions on the trained model
        X: (m, n)  m(# training e.g.)  n(# features)

        return y: (m, )
        """
        # reshape
        X = X.transpose()  # (n, m)
        w = self.w.reshape(-1, 1)  # (n, 1)
        b = self.b

        y = sigmoid(np.dot(w.T, X) + b)
        y = y.squeeze()
        y[y < 0] = -1
        y[y >= 0] = 1
        return y


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



"""
Provide model for different base functions.
This file only use numpy and matplotlib package.
"""

import numpy as np
import matplotlib.pyplot as plt


def power(x_train, y_train, n=2, learning_rate=0.0005, epochs=1000, l2=0, Print=False):
    """
    Power base function.
    This target function is: y = b + w1 * x^1 + w2 * x^2 + ...
    also y = b + np.dot(w.T, x)

    :param x_train: np.ndarray
    :param y_train: np.ndarray
    :param learning_rate:
    :param epochs:

    :return: a trained model (as a function), trained by x_train and y_train
    """

    # get the number of train e.g.
    m = x_train.shape[0]

    # # set some hyper-parameter here.
    # # the max value of power
    # n = 2

    # set and initialize parameters here
    # intercept
    # b = np.float64(0)
    b = np.float64(-10)
    # weights
    w = np.float64(np.random.randn(n, 1))

    # convert the x_train matrix to a design matrix
    X = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        X[i, :] = x_train ** (i + 1)
    X = np.float64(X)
    Y = np.float64(np.reshape(y_train, newshape=(1, m)))

    # if plot of the training process is needed
    costs = []
    Z = b + np.dot(w.T, X)
    dZ = Z - Y
    cost = 0.5/m * np.dot(dZ, dZ.T)
    costs.append(cost.squeeze())

    # train on the dataset
    for epoch in range(epochs):
        # compute the gradient of cost on w

        Z = b + np.dot(w.T, X)

        dZ = Z - Y
        dw = 1./m * np.dot(X, dZ.T)
        db = 1./m * np.squeeze(np.sum(dZ))

        # update the parameters, for w, I also set "weight decay"
        w -= learning_rate * dw + l2 * w
        b -= learning_rate * db

        cost = np.squeeze(0.5/m * np.dot(dZ, dZ.T))
        costs.append(cost)
        if Print == True and epoch % 25 == 0:
            print("Cost after " + str(epoch) + " iterations " + ": " + str(cost))

    # plot the costs
    if Print == True:
        plt.plot(costs)
        plt.show()

    # store the parameters in a python dict
    params = {'b':b, 'w':w, 'n':n}

    def pred(x):
        """

        :param x:
        :return:
        """
        assert type(x) is np.ndarray

        m = x.shape[0]

        # convert the x_train matrix to a design matrix
        X = np.zeros((n, m))
        for i in range(n):
            X[i, :] = x ** (i + 1)

        # to predict
        Y = b + np.dot(w.T, X)

        # print(Y.squeeze())

        return Y.squeeze()

    return pred


def gaussian(x_train, y_train, n=2, learning_rate=0.01, epochs=10000, l2=0, Print=False):
    """
    use gaussian base for generalized linear regression, params(w, mu, s)
    :param x_train:
    :param y_train:
    :param learning_rate:
    :param epochs:
    :param l2:
    :param Print:
    :return:
    """

    assert type(x_train) is np.ndarray

    # the number of training e.g.
    m = x_train.shape[0]

    # # a hyper-parameter, the dim of gauss
    # n = 2

    # set and init some params here
    # the weights
    w = np.random.randn(n, 1)
    # the intercept
    b = np.float64(0)
    # the means of gaussian
    mu = np.random.randn(n, 1)
    # I think different mus should be init differently
    for i in range(n):
        mu[i, 0] += i * 40

    # take care, if s -> 0, the algorithm go wrong,
    while True:
        flag = True
        s = np.random.randn(n, 1)

        conditions = np.int32(np.abs(s) < 1e-5)
        # if some element of s is too close to 0, re-init s
        if np.sum(conditions) >= 1:
            flag = False
        if flag is True:
            break


    # convert the x_train matrix to a design matrix
    # X.shape=(n,m)    Y.shape=(1,m)
    X = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        X[i, :] = x_train
    # X = np.float64(X)
    Y = np.reshape(y_train, newshape=(1, m))

    # to store costs in a list
    costs = []

    # train the parameters
    for epoch in range(epochs):
        # forward prop
        Z = (X - mu) / s
        A = np.exp(- (Z * Z) / 2)
        Y_hat = np.dot(w.T, A) + b

        # compute cost
        cost = 1./ (2*m) * np.dot(Y_hat - Y, (Y_hat - Y).T)
        cost = cost.squeeze()

        # backprop
        dY = 1./m * (Y_hat - Y)
        dw = np.dot(A, dY.T)
        db = np.squeeze(np.sum(dY))
        dA = w * dY
        dZ = dA * A * (-Z)
        dmu = 1./m * np.sum(dZ * (- 1/s), axis=1, keepdims=True)
        ds = 1./m * np.sum(dZ * (-(X - mu) / s**2 ), axis=1, keepdims=True)

        # update the params
        w -= learning_rate * dw + l2 * w
        b -= learning_rate * db + l2 * b
        mu -= learning_rate * dmu + l2 * mu
        s -= learning_rate * ds + l2 * s

        # store the cost
        costs.append(cost)
        if Print is True and epoch % 10000 == 0:
            print("Cost after " + str(epoch) + " iterations " + ": " + str(cost))

        pass

    # print the curve of costs
    if Print is True:
        plt.plot(costs)
        plt.show()

    # def the trained model
    def model(x):
        """
        a trained gaussian base model
        :param x:
        :return:
        """
        assert type(x) is np.ndarray

        m = x.shape[0]

        X = np.zeros(shape=(n, m))
        for i in range(n):
            X[i, :] = x

        # forward prop
        Z = (X - mu) / s
        A = np.exp(- (Z * Z) / 2)
        Y_hat = np.dot(w.T, A) + b

        return Y_hat.squeeze()

    return model


def sigmoid_base(x_train, y_train, n=2, learning_rate=0.001, epochs=10000, l2=0, Print=False):
    """

    :param x_train:
    :param y_train:
    :param n:
    :param learning_rate:
    :param epochs:
    :param l2:
    :param Print:
    :return:
    """
    assert type(x_train) is np.ndarray
    assert type(y_train) is np.ndarray

    # def a sigmoid function here, np version
    def sigmoid(z):
        a = 1. / (1 + np.exp(-z))
        return a

    # get the number of e.g.
    m = x_train.shape[0]

    # set and init some params
    # the weights vector
    w = np.random.randn(n, 1)
    # the intercept
    b = np.float64(0)
    # the means of sigmoids
    mu = np.random.randn(n, 1)
    # I think different mus should be init differently
    for i in range(n):
        mu[i, 0] += i * 100 / n
    # take care, if s -> 0, the algorithm go wrong,
    while True:
        flag = True
        s = np.random.randn(n, 1)
        conditions = np.int32(np.abs(s) < 1e-5)
        # if some element of s is too close to 0, re-init s
        if np.sum(conditions) >= 1:
            flag = False
        if flag is True:
            break

    # convert the x_train matrix to a design matrix
    # X.shape=(n,m)    Y.shape=(1,m)
    X = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        X[i, :] = x_train
    # X = np.float64(X)
    Y = np.reshape(y_train, newshape=(1, m))

    # to store costs in a list
    costs = []

    # train the params
    for epoch in range(epochs):
        # forward prop
        Z = (X - mu) / s
        A = sigmoid(Z)
        Y_hat = b + np.dot(w.T, A)

        # compute cost
        cost = 1./(2*m) * np.dot(Y_hat - Y, (Y_hat - Y).T).squeeze()
        
        # back-prop
        dY_hat = 1./m * (Y_hat - Y)
        dw = np.dot(A, dY_hat.T)
        db = np.squeeze(np.sum(dY_hat))
        dA = w * dY_hat
        dZ = dA * A * (1 - A)
        dmu = 1./m * np.sum(dZ * (-1./s), axis=1, keepdims=True)
        ds = 1./m * np.sum(dZ * (-(X - mu) / s ** 2), axis=1, keepdims=True)

        # update params
        w -= learning_rate * dw + l2 * w
        b -= learning_rate * db + l2 * b
        mu -= learning_rate * dmu + l2 * mu
        s -= learning_rate * ds + l2 * s

        # store cost of every iteration
        costs.append(cost)

        if Print is True and epoch % 10000 == 0:
            print("Cost after " + str(epoch) + " iterations " + ": " + str(cost))

        pass

    if Print is True:
        plt.plot(costs)
        plt.title("Costs")
        plt.show()
        pass

    def model(x):
        assert type(x) is np.ndarray

        m = x.shape[0]

        X = np.zeros((n, m), dtype=np.float64)
        for i in range(n):
            X[i, :] = x

        # forward-prop
        Z = (X - mu) / s
        A = sigmoid(Z)
        Y_hat = b + np.dot(w.T, A)

        return Y_hat.squeeze()

    return model


def mix(x_train, y_train, learning_rate=0.0005, epochs=50000, l2=0, Print=True):
    """
    y = w0 + w1*x + w2*x + c*sin(dx + b)
    :param x_train:
    :param y_train:
    :param learning_rate:
    :param epochs:
    :param l2:
    :param Print:
    :return:
    """
    assert type(x_train) is np.ndarray
    assert type(y_train) is np.ndarray

    X, Y = x_train, y_train

    # set a flag for sin
    no_sin = False

    # get the number is e.g.
    m = x_train.shape[0]

    # set some params and init them
    w0 = np.float64(-30)
    w1 = np.random.randn()
    w2 = np.float64(1e-2)
    c = np.random.randn()
    d = np.float64(np.pi/20)
    b = np.float64(2*np.pi/5)

    # to store costs
    costs = []

    for epoch in range(epochs):
        # forward-prop
        Z = d * X + b
        A = np.sin(Z)
        Y_hat = w0 + w1 * X + w2 * X * X + c * A * (1 - int(no_sin))

        # compute cost
        cost = 0.5/m * np.dot(Y_hat - Y, Y_hat - Y)

        # back-prop
        dY = 1./m * (Y_hat - Y)
        if no_sin is False:
            dA = dY * c
            dc = np.mean(dY * A)
            dZ = dA * np.cos(Z)
            dd = np.mean(dZ * X)
            db = np.mean(dZ)
        dw0 = np.mean(dY)
        dw1 = np.mean(dY * X)
        dw2 = np.mean(dY * X * X)

        # update params
        w0 -= learning_rate * dw0
        w1 -= learning_rate * dw1 + l2 * w1
        # beta = 1e-3
        # w2 -= beta * learning_rate * dw2 + l2 * w2
        w2 = 0.0013
        if no_sin is False:
            alpha = 1e3
            c -= alpha * learning_rate * dc
            # d -= alpha * learning_rate * dd
            d = np.pi / 19
            b -= alpha * learning_rate * db

        # store the cost
        costs.append(cost)
        if Print is True and epoch % 10000 == 0:
            print("Cost after " + str(epoch) + " iterations " + ": " + str(cost))

        pass

    if Print is True:
        plt.plot(costs)
        plt.title("Costs")
        plt.show()


    def model(x):
        assert type(x) is np.ndarray

        Z = d * x + b
        A = np.sin(Z)

        Y_hat = w0 + w1 * x + w2 * x * x + c * A * (1 - int(no_sin))
        return Y_hat

    return model


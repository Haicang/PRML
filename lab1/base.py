"""
Provide model for different base functions.
This file only use numpy and matplotlib package.
"""

import numpy as np
import matplotlib.pyplot as plt


def power(x_train, y_train, learning_rate=0.0005, epochs=1000, l2=0, Print=False):
    """
    Power base function.
    This target function is: y = b + w1 * x^1 + w2 * x^2 + ...
    also y = b + np.dot(w.T, x)

    :param x_train:
    :param y_train:
    :param learning_rate:
    :param epochs:

    :return:
    """

    # get the number of train e.g.
    m = x_train.shape[0]

    # set some hyper-parameter here.
    # the max value of power
    n = 1

    # set and initialize parameters here
    # intercept
    b = 0
    # weights
    w = np.random.randn(n, 1)

    # convert the x_train matrix to a design matrix
    X = np.zeros((n, m))
    for i in range(n):
        X[i, :] = x_train ** (i + 1)
    Y = np.reshape(y_train, newshape=(1, m))

    # if plot of the training process is needed
    costs = []
    Z = b + np.dot(w.T, X)
    dZ = Z - Y
    cost = 0.5/m * np.dot(dZ, dZ.T)
    costs.append(np.float32(cost.squeeze()))

    # train on the dataset
    for epoch in range(epochs):
        # compute the gradient of cost on w
        # tmp.shape = (1, m)
        # tmp = b + np.dot(w.T, X) - Y
        Z = b + np.dot(w.T, X)
        # print(X)
        # print(b)
        # print(w)
        # print(Z)
        dZ = Z - Y
        dw = 1./m * np.dot(X, dZ.T)
        db = 1./m * np.squeeze(np.sum(dZ))
        # print("w\n", w)
        # print("b\n", b)
        # print("dw\n", dw)
        # print("db\n", db)
        # print("X\n", X)
        # print("Y\n", Y)
        # print("Z\n", Z)
        # print("dZ\n", dZ)
        # print("\n\n")

        # update the parameters, for w, I also set "weight decay"
        w -= learning_rate * dw + l2 * w
        b -= learning_rate * db

        cost = np.float32(np.squeeze(0.5/m * np.dot(dZ, dZ.T)))
        costs.append(cost)
        if Print == True and epoch % 10000 == 0:
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

        return Y.squeeze()

    return pred


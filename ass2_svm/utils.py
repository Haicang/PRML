"""
author: Haicang

Some utility functions for svm.py, but not for algorithms

Some functions for visualization are also here
"""

# If a float x is |x| < delta, I regard it's 0
# using for decrease support vecs
DELTA = 1e-5


# Use to raise Exception
class NotOptimal(ValueError):
    def __init__(self, message='Quadratic Optimazation Fail'):
        self.message = message


# Visualization
import numpy as np 
import matplotlib.pyplot as plt
from threading import Thread


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def show_models(models, titles, X, t, savename=None):
    n = len(models)

    fig, sub = plt.subplots(1, n)
    fig.set_size_inches(5 * n, 5)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=t, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('X0')
        ax.set_ylabel('X1')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    if savename != None:
        plt.savefig(savename)
    plt.show()


def show_models_multithread(models, titles, X, t, savename=None):
    """
    A parallel version to draw the contour
    This function is troublesome
    """

    n = len(models)

    fig, sub = plt.subplots(1, n)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    Threads = []

    def draw_sub(ax, clf, title):
        
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=t, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        Threads.append(Thread(target=draw_sub, args=(ax, clf, title), name=title))

    for t in Threads:
        t.start()

    for t in Threads:
        t.join()

    if savename != None:
        plt.savefig(savename)
    plt.show()


# data loading
def load_data_to_fit(fname):
    """
    return X, t
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
        data = np.array(data)
        
        X = data[:, :2]
        t = data[:, 2]
        return X, t

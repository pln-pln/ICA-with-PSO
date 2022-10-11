# for the animation part
from __future__ import print_function


# for PSO part
import random
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import fitnessfunction

# define the laplacian rand variables

s1 = np.random.rand(1000)
#x = np.linspace(laplace.ppf(0.01), laplace.ppf(0.99), 100)

s2 = np.random.rand(1000)


#source signal
S = np.array([s1, s2]).T

#mixing matrices
A = np.array(([0.96, -0.28],[0.28, 0.96]))

# mixed signal/matrix
X = S.dot(A).T



# preprocessing
def center(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered = x - mean
    return centered, mean

def covariance(x):
    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x-mean
    return (m.dot(m.T))/n

def whiten(x):
    covar_matrix = covariance(x)

    # single value decomposition
    u, s, v = np.linalg.svd(covar_matrix)

    # diagonal matrix of eigenvalues
    d = np.diag(1.0/np.sqrt(s))

    # whitening matrix
    whiteM = np.dot(u, np.dot(d, u.T))

    # project onto whitening matrix (?)
    Xw = np.dot(whiteM, x)

    test = np.dot(Xw, Xw.T)
    return Xw

Xc, meanX = center(X)

Xw = whiten(Xc)


def fastIca(signals, alpha=1, thresh=1e-8, iterations=5000):
    m, n = signals.shape

    # Initialize random weights
    W = np.random.rand(m, m)

    for c in range(m):
        w = W[c, :].copy().reshape(m, 1)
        w = w / np.sqrt((w ** 2).sum())  #normalize

        i = 0
        lim = 100
        while ((lim > thresh) & (i < iterations)):
            # Dot product of weight and signal
            ws = np.dot(w.T, signals)

            # Pass w*s into contrast function g
            wg = np.tanh(ws * alpha).T

            # Pass w*s into g prime
            wg_ = (1 - np.square(np.tanh(ws))) * alpha

            # Update weights
            wNew = (signals * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()

            # Decorrelate weights
            wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
            wNew = wNew / np.sqrt((wNew ** 2).sum())

            # Calculate limit condition
            lim = np.abs(np.abs((wNew * w).sum()) - 1)

            # Update weights
            w = wNew

            # Update counter
            i += 1

        W[c, :] = w.T
    return W
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import moment

def fitness(x):

    mean = np.mean(x, keepdims=True) # Calculate the mean
    var = np.var(x, keepdims=True) # Calculate the variance
    skew = moment(x, moment=3)

    kurt = kurtosis(x)

    J = (skew**2)/12 + (kurt**2)/48

    return J

A = np.array(([0.96, -0.28,0.1],[0.28, 0.96, 0.1], [0.96, 0.1, 0.28]))
ff = fitness(A[:, 0])
print(ff)


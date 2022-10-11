# for the animation part
from __future__ import print_function


# for PSO part
import random
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import numpy.linalg as la
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












plt.style.use('bmh')

low_bound = -1  # lower bound value to be used in axis limits
up_bound = 1    # upper bound value to be used in axis limits

limits = ([low_bound, up_bound],  # limits for x axis
          [low_bound, up_bound])  # limits for y axis

# limits is a tuple that allows us to choose boundaries of the axes conveniently from the corresponding lists it holds
# below are the boundaries

x_lo = limits[0][0]
x_up = limits[0][1]
y_lo = limits[1][0]
y_up = limits[1][1]

negent = fitnessfunction.fitness  # the fitness function is chosen to be ackley and is represented with f.

def f(x, y):
    w = np.array([x, y])
    w = w / np.linalg.norm(w)
    return negent(np.dot(w, Xw))

# ----main----

n_iterations = 150

def run_pso_ica(signals, alpha=1, thresh=1e-8,n_particles = 7, omega=0.5, phi_p=2, phi_g=2):
    """
    :param omega: particle weight (inertial)
    :param phi_p: particle best weight
    :param phi_g: global weight
    :return:
    """

    global w_best_p_global, z_p_best_global, \
            w_particles, z_particles, \
            u_particles

    m, n = signals.shape


    W = np.zeros((m,m))

    for row in range(m):

        x_particles = np.zeros((n_iterations, n_particles))
        # creates an array of n_iterations (as number) arrays containing n_particles zeros each.
        x_particles[0, :] = np.random.uniform(low_bound, up_bound, size=n_particles)
        # Draw samples from a uniform distribution. Assigns them to previously created arrays of location
        #print("x_particles initial - ",x_particles)


            # initialize y positions of particles
        y_particles = np.zeros((n_iterations, n_particles))
        y_particles[0, :] = np.random.uniform(low_bound, up_bound, size=n_particles)
        #print("y_particles initial - ", y_particles)


        # initialize personal best of particles
        x_best_particles = np.copy(x_particles[0, :])
        y_best_particles = np.copy(y_particles[0, :])

        # z axis will show the calculated fitness function
        z_particles = np.zeros((n_iterations, n_particles))

        for j in range(n_particles):
            z_particles[0, j] = f(x_particles[0, j], y_particles[0, j])

        # define the global best value of fitness and corresponding best index
        z_best_global = np.min(z_particles[0, :])
        index_best_global = np.argmin(z_particles[0, :])

        # define global best x and y values
        x_best_p_global = x_particles[0, index_best_global]
        y_best_p_global = y_particles[0, index_best_global]
        wp = np.array([x_best_p_global, y_best_p_global])

        # initialize velocity
        velocity_lo = low_bound - up_bound
        velocity_up = up_bound - low_bound

        v_max = 0.07
        u_max = 0.07

        u_particles = np.zeros((n_iterations, n_particles))
        u_particles[0, :] = 0.1*np.random.uniform(velocity_lo, velocity_up, size=n_particles)

        v_particles = np.zeros((n_iterations, n_particles))
        v_particles[0, :] = 0.1*np.random.uniform(velocity_lo, velocity_up, size=n_particles)

        # PSO starts here

        iteration = 1

        while iteration <= n_iterations - 1:


            for i in range(n_particles):
                x_p = x_particles[iteration - 1, i]  # x location of particles in that iteration
                y_p = y_particles[iteration - 1, i]  # y loc of particles at that iteration

                u_p = u_particles[iteration - 1, i]
                v_p = v_particles[iteration - 1, i]

                x_best_p = x_best_particles[i]  # best personal x
                y_best_p = y_best_particles[i]  # best personal y

                # R1 and R2 constants that were diagonal matrices in the article
                r_p = np.random.uniform(0, 1)
                r_g = np.random.uniform(0, 1)

                # update velocity
                u_p_new = omega * u_p + phi_p * r_p * (x_best_p - x_p) + \
                          phi_g * r_g * (x_best_p_global - x_p)

                v_p_new = omega * v_p + phi_p * r_p * (y_best_p - y_p) + \
                          phi_g * r_g * (y_best_p_global - y_p)

                # !!! Velocity control !!!

                while not (-v_max <= u_p_new <= v_max):
                    u_p_new = 0.9 * u_p_new
                while not (-v_max <= v_p_new <= v_max):
                    v_p_new = 0.9 * v_p_new

                #print("u, v: ", [u_p_new, v_p_new])
                # update the positions as particles move
                x_p_new = x_p + u_p_new
                y_p_new = y_p + v_p_new

                # ignore new positions if they are out of domain
                if not ((low_bound <= x_p_new <= up_bound) and (low_bound <= y_p_new <= up_bound)):
                    x_p_new = x_p
                    y_p_new = y_p



                # update the matrices
                x_particles[iteration, i] = x_p_new
                y_particles[iteration, i] = y_p_new

                u_particles[iteration, i] = u_p_new
                v_particles[iteration, i] = v_p_new

                # evaluate fitness
                z_p_new = f(x_p_new, y_p_new)
                #print("z_p_new: ", z_p_new)
                z_p_best = f(x_best_p, y_best_p)
                #print("z_p_best: ", z_p_best)
                z_particles[iteration, i] = z_p_new

                if z_p_new > z_p_best:  # check to update personal best
                    x_best_particles[i] = x_p_new  # update the matrix that was initialized to 0's
                    y_best_particles[i] = y_p_new

                    z_p_best_global = f(x_best_p_global, y_best_p_global)
                    print("z_p_best_global: ", z_p_best_global)

                    if z_p_new > z_p_best_global:  # check to update global bests
                        x_best_p_global = x_p_new
                        y_best_p_global = y_p_new
                        wp = np.array([x_best_p_global, y_best_p_global])

            wp = wp / np.linalg.norm(wp)
            print([x_best_p_global, y_best_p_global])
            # increase iteration number to keep iterating
            iteration += 1

            if row == 1:
                b = wp
                v1 = W[0]
                nominator = np.dot(v1.T, b)
                denominator = np.dot(v1.T, v1)
                proj = np.dot(nominator/denominator, v1)
                orth = b - proj
                wp = orth / np.linalg.norm(orth)

        W[row] = np.array(wp)
        """if row == 0:
            W[row] = np.array([x_best_p_global, y_best_p_global])
        else:
            W[row] = np.array([y_best_p_global, x_best_p_global])"""
        # plot convergence
        z_particles_best_hist = np.min(z_particles, axis=1)
        z_particles_worst_hist = np.max(z_particles, axis=1)
    print(W)
    return W


def orthogonalize(U, eps=1e-15):

    n = len(U[0])
    # numpy can readily reference rows using indices, but referencing full rows is a little
    # dirty. So, work with transpose(U)
    V = U.T
    for i in range(n):
        prev_basis = V[0:i]  # orthonormal basis before V[i]
        coeff_vec = np.dot(prev_basis, V[i].T)  # each entry is np.dot(V[j], V[i]) for all j < i
        # subtract projections of V[i] onto already determined basis V[0:i]
        V[i] -= np.dot(coeff_vec, prev_basis).T
        if la.norm(V[i]) < eps:
            V[i][V[i] < eps] = 0.  # set the small entries to 0
        else:
            V[i] /= la.norm(V[i])
    return V.T


W = run_pso_ica(Xw)
print(W)
W = orthogonalize(W)
print(W)

unMixed = Xw.T.dot(W.T)
unMixed = (unMixed.T - meanX).T


# scatter plot of source
fig3, ax3 = plt.subplots(1, 2, figsize=[18, 10])
ax3[0].scatter(S.T[0], S.T[1])
ax3[0].set_xlim([-0.25, 1.5])
ax3[0].set_title('source', fontsize=25)
# scatter plot of mixed matrix
ax3[1].scatter(X[0], X[1])
ax3[1].set_xlim([-0.25, 1.5])
ax3[1].set_title('mixed', fontsize=25)
# whitened
"""fig2, ax2 = plt.subplots(1, 2, figsize=[18, 5], sharex='row')
ax2[0].scatter(X[0], X[1])
ax2[0].set_title('not whitened', fontsize=25)
ax2[0].set_xlim([-0.25, 1.5])

ax2[1].scatter(Xw[0], Xw[1])
ax2[1].set_title('whitened', fontsize=25)
ax2[1].set_xlim([-5, 5])
"""

# scatter plot of source
fig6, ax6 = plt.subplots(1, 2, figsize=[18, 5])
ax6[0].scatter(S.T[0], S.T[1])
ax6[0].set_xlim([-5, 5])
ax6[0].set_title('source', fontsize=25)
# scatter plot of mixed matrix
ax6[1].scatter(unMixed.T[0], unMixed.T[1])
ax6[1].set_xlim([-5, 5])
ax6[1].set_title('recovered', fontsize=25)
plt.show()






























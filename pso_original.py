# for the animation part
from __future__ import print_function
import ipywidgets as widgets
from IPython.display import display, HTML

# for PSO part
import random
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotPSO import plotPSO_2D
import fitnessfuncs




plt.style.use('bmh')

low_bound = -2  # lower bound value to be used in axis limits
up_bound = 2    # upper bound value to be used in axis limits

limits = ([low_bound, up_bound],  # limits for x axis
          [low_bound, up_bound])  # limits for y axis

# limits is a tuple that allows us to choose boundaries of the axes conveniently from the corresponding lists it holds
# below are the boundaries

x_lo = limits[0][0]
x_up = limits[0][1]
y_lo = limits[1][0]
y_up = limits[1][1]

f = fitnessfuncs.ackley  # the fitness function is chosen to be ackley and is represented with f.


# ----main----

n_iterations = 50

def run_pso(n_particles=20, omega=0.9, phi_p=2, phi_g=2):
    """
    :param omega: particle weight (inertial)
    :param phi_p: particle best weight
    :param phi_g: global weight
    :return:
    """

    global x_best_p_global, y_best_p_global, z_p_best_global, \
            x_particles, y_particles, z_particles, \
            u_particles, v_particles


    # initialize x positions of particles.

    x_particles = np.zeros((n_iterations, n_particles))
    # creates an array of n_iterations (as number) arrays containing n_particles zeros each.
    x_particles[0, :] = np.random.uniform(low_bound, up_bound, size=n_particles)
    # Draw samples from a uniform distribution. Assigns them to previously created arrays of location

    # initialize y positions of particles
    y_particles = np.zeros((n_iterations, n_particles))
    y_particles[0, :] = np.random.uniform(low_bound, up_bound, size=n_particles)

    # initialize personal best of particles
    x_best_particles = np.copy(x_particles[0, :])
    y_best_particles = np.copy(y_particles[0, :])

    # z axis will show the calculated fitness function
    z_particles = np.zeros((n_iterations, n_particles))

    for j in range(n_particles):
        z_particles[0, j] = f((x_particles[0, j], y_particles[0, j]))

    # define the global best value of fitness and corresponding best index
    z_best_global = np.min(z_particles[0, :])
    index_best_global = np.argmin(z_particles[0, :])

    # define global best x and y values
    x_best_p_global = x_particles[0, index_best_global]
    y_best_p_global = y_particles[0, index_best_global]

    # initialize velocity
    velocity_lo = low_bound-up_bound
    velocity_up = up_bound-low_bound

    v_max = 0.07
    u_max = 0.07

    u_particles = np.zeros((n_iterations, n_particles))
    u_particles[0, :] = 0.1*np.random.uniform(velocity_lo, velocity_up, size=n_particles)

    v_particles = np.zeros((n_iterations, n_particles))
    v_particles[0, :] = 0.1 * np.random.uniform(velocity_lo, velocity_up, size=n_particles)

    # PSO starts here

    iteration = 1

    while iteration <= n_iterations-1:

        for i in range(n_particles):
            x_p = x_particles[iteration-1, i]  # x location of particles in that iteration
            y_p = y_particles[iteration-1, i]  # y loc of particles at that iteration

            u_p = u_particles[iteration-1, i]
            v_p = v_particles[iteration-1, i]

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
            z_p_new = f((x_p_new, y_p_new))
            z_p_best = f((x_best_p, y_best_p))

            z_particles[iteration, i] = z_p_new



            if z_p_new < z_p_best:  # check to update personal best
                x_best_particles[i] = x_p_new  # update the matrix that was initialized to 0's
                y_best_particles[i] = y_p_new

                z_p_best_global = f([x_best_p_global, y_best_p_global])

                if z_p_new < z_p_best_global:  # check to update global bests
                    x_best_p_global = x_p_new
                    y_best_p_global = y_p_new

        # increase iteration number to keep iterating
        iteration += 1

    # plot convergence
    z_particles_best_hist = np.min(z_particles, axis=1)
    z_particles_worst_hist = np.max(z_particles, axis=1)

    print(z_particles_best_hist)
    print(z_particles_worst_hist)
    z_best_global = np.min(z_particles)
    index_best_global = np.argmin(z_particles)



    return (z_particles_best_hist, z_particles_worst_hist)

run_pso()

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))

# Grid points
x_lo = limits[0][0]
x_up = limits[0][1]
y_lo = limits[1][0]
y_up = limits[1][1]

assert x_lo < x_up, "Unbound x limits, the first value of the list needs to be higher"
assert y_lo < y_up, "Unbound x limits, the first value of the list needs to be higher"

n_points = 100

x = np.linspace(x_lo, x_up, n_points)  # x coordinates of the grid
y = np.linspace(y_lo, y_up, n_points)  # y coordinates of the grid

XX, YY = np.meshgrid(x, y)
ZZ = np.zeros_like(XX)

for i in range(n_points):
    for j in range(n_points):
        ZZ[i, j] = f((XX[i, j], YY[i, j]))

# Limits of the function being plotted
ax1.plot((0, n_iterations), (np.min(ZZ), np.min(ZZ)), '--g', label="min$f(x)$")
ax1.plot((0, n_iterations), (np.max(ZZ), np.max(ZZ)), '--r', label="max$f(x)$")

ax1.plot(np.arange(n_iterations),run_pso()[0],'b',  label="$p_{best}$")
ax1.plot(np.arange(n_iterations),run_pso()[1],'k', label="$p_{worst}$")

ax1.set_xlim((0,n_iterations))

ax1.set_ylabel('$f(x)$')
ax1.set_xlabel('$i$ (iteration)')
ax1.set_title('Convergence')

ax1.legend()

plt.show()


""" ANIMATION 

def plotPSO_iter(i=0):
    #visualization of particles and fitness function

    fig, (ax1, ax2) = plotPSO_2D(f, limits,
                                 particles_xy=(x_particles[i,:], y_particles[i, :]),
                                 particles_uv=(u_particles[i,:], v_particles[i, :]))


w_arg_PSO = widgets.interact_manual(run_pso,
                            n_particles=(2,50),
                            omega=(0,1,0.001),
                            phi_p=(0,1,0.001),
                            phi_g=(0,1,0.001),
                            continuous_update=False)

w_viz_PSO = widgets.interact_manual(plotPSO_iter, i=(0,n_iterations-1), continuous_update=False)

"""
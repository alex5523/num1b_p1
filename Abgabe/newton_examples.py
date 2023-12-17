#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 12:27:07 2023

@author: alexandrajohansen
"""

import numpy as np
from newton_more_variables import newton_method1
from scipy.optimize import fsolve
from newton_plots import plot_intersection


#############################
# Example functions
#############################

# Example 1: Hyperbola and circle
# Function
def f1(x):
    m1 = [0.0,0.0]
    m2 = [0.0,0.0]
    r1 = np.sqrt(2)
    r2 = np.sqrt(1)
    return np.array([(x[0] - m1[0])**2 + (x[1] - m1[1])**2 - r1**2,
                     (x[0] - m2[0])**2 - (x[1] - m2[1])**2 - r2**2])

# Jacobian of function
def J1(x):
    return np.array([[2*x[0], 2*x[1]],
                     [2*x[0], -2*x[1]]])

# Initial guess
x_init = [1.0, 1.0]

# MYSOLVER
x, k_end = newton_method1(f1, J1, x_init)

#print("EX1: A circle and a hyperbola", x)
#print("Newton Method Solution:", x)
#print("Number of Iterations:", k_end)

# FSOLVER
sol = fsolve(f1, x_init)  # Löse nichtlineares Gleichungssystem
#print("fsolve =", sol)



# Example 2: 2 circles
# Function
def f2(x):
    m1 = [0.0,0.0]
    m2 = [0.0,5.0]
    r1 = 3
    r2 = 3
    return np.array([(x[0] - m1[0])**2 + (x[1] - m1[1])**2 - r1**2,
                     (x[0] - m2[0])**2 + (x[1] - m2[1])**2 - r2**2])

# Jacobian of function
def J2(x):
    return np.array([[2*x[0], 2*x[1]],
                     [2*x[0], 2*x[1] - 10]])

# Initial guess
x_init = [1.0, 4.0]

# MYSOLVER
x, k_end = newton_method1(f2, J2, x_init)

#print("\nEx2: Two circles", x)
#print("Newton Method Solution:", x)
#print("Number of Iterations:", k_end)

# FSOLVER
sol = fsolve(f2, x_init)  # Löse nichtlineares Gleichungssystem
#print("fsolve=", sol)


def ex2():
    x1_vals = np.linspace(-6, 6, 400)
    x2_vals = np.linspace(-3, 8, 400)
    plot_intersection(f2, J2, [1.0, 4.0], 'Intersection of Two Circles', x1_vals, x2_vals)
    print("\nEx2: Two circles", x)
    print("Newton Method Solution:", x)
    print("Number of Iterations:", k_end)
    return






# Example 3: Three spheres
# Function
def f3(x):
    m1 = [2.0, 0.0, -1.0]
    m2 = [1.0, 2.0, 0.0]
    m3 = [0.0, 0.0, -2.0]
    r1 = 2.0
    r2 = 2.0
    r3 = 2.0
    
    return np.array([
        (x[0] - m1[0])**2 + (x[1] - m1[1])**2 + (x[2] - m1[2])**2 - r1**2,
        (x[0] - m2[0])**2 + (x[1] - m2[1])**2 + (x[2] - m2[2])**2 - r2**2,
        (x[0] - m3[0])**2 + (x[1] - m3[1])**2 + (x[2] - m3[2])**2 - r3**2
    ])

# Jacobian of function
def J3(x):
    return np.array([
        [2 * (x[0] - 2.0), 2 * (x[1] - 0.0), 2 * (x[2] - 1.0)],
        [2 * (x[0] - 1.0), 2 * (x[1] - 2.0), 2 * (x[2] - 0.0)],
        [2 * (x[0] - 0.0), 2 * (x[1] - 0.0), 2 * (x[2] - (-2.0))]
    ])

# Initial guess
#x_init = [1.0, 0.0, -1.0]
x_init = [-1.0, 0.0, 1.0] # am schnellsten!
#x_init = [-1.0, 0.0, -1.0]


# MYSOLVER
x, k_end = newton_method1(f3, J3, x_init)

#print("\nEX3: Three spheres", x)
#print("Newton Method Solution:", x)
#print("Number of Iterations:", k_end)

# FSOLVER
sol = fsolve(f3, x_init)  # Löse nichtlineares Gleichungssystem
#print("fsolve=", sol)



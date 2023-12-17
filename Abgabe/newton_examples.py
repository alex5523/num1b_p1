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

# Example 1: Intersection of a circle and a hyperbola
def ex1(x_init):
    
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
    x_init = x_init

    # MYSOLVER
    x, k_end = newton_method1(f1, J1, x_init)
    
    print("Example 1: Intersection of a circle and a hyperbola")
    if x is not None:
        sol = fsolve(f1, x_init)  # Löse nichtlineares Gleichungssystem
        print("My Newton Method Solution:\n", x)
        print("Number of Iterations:", k_end)
        print("fsolve:\n", sol)
        print("Mysolution = fsolve ? ", np.allclose(sol, x))
    else:
        print("No solution to this LSE!")
        
    

    
    # Visualize the solution
    x1_vals = np.linspace(-3, 3, 400)
    x2_vals = np.linspace(-3, 3, 400)
    plot_intersection(f1, J1, x_init, 'Intersection of a Circle and a Hyperbola', x1_vals, x2_vals)

    return

# Debugging test example
#x_init = [1.0, 1.0] # converges
#x_init = [1.0, 1.0, 1.0] # wont cut dimension checks
#x_init = np.array([[1.0, 1.0], [1.0, 1.0]]) # wont cut dimension checks
#x_init = [1.0, 0.0] # Jacobian singular
#ex1(x_init)



# Example 2: Two intersecting circles
def ex2(x_init):
    
    # Function
    def f2(x):
        m1 = [0.0, 0.0]
        m2 = [0.0, 5.0]
        r1 = 3
        r2 = 3
        return np.array([(x[0] - m1[0])**2 + (x[1] - m1[1])**2 - r1**2,
                         (x[0] - m2[0])**2 + (x[1] - m2[1])**2 - r2**2])

    # Jacobian of function
    def J2(x):
        return np.array([[2*x[0], 2*x[1]],
                         [2*x[0], 2*x[1] - 10]])

    # Initial guess
    #x_init = [1.0, 4.0] # converges

    # MYSOLVER
    x, k_end = newton_method1(f2, J2, x_init)    
    
    print("Example 2: Two intersecting circles")
    if x is not None:
        sol = fsolve(f2, x_init)
        print("My Newton Method Solution:\n", x)
        print("fsolve:\n", sol)
        print("Mysolution = fsolve ? ", np.allclose(sol, x))
        print("Number of Iterations:", k_end)
    else:
        print("No solution to this LSE!")


    # Visualise the solution
    x1_vals = np.linspace(-6, 6, 400)
    x2_vals = np.linspace(-3, 8, 400)
    plot_intersection(f2, J2, x_init, 'Intersection of Two Circles', x1_vals, x2_vals)       
    return

#x_init = [1.0, 1.0]
#ex2(x_init)

# Example 3: Three intersecting spheres
def ex3(x_init):
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
        
    # MYSOLVER
    x, k_end = newton_method1(f3, J3, x_init)
    
    # FSOLVER
    sol = fsolve(f3, x_init)  # Löse nichtlineares Gleichungssystem
    
    print("Example 3: Three intersecting spheres")
    if x is not None:
        print("My Newton Method Solution:\n", x)
        print("Number of Iterations:", k_end)
        print("fsolve:\n", sol)
        print("Mysolution = fsolve ? ", np.allclose(sol, x))
    else:
        print("No solution to this LSE!")

    return



# Example 4: Two non intersecting circles
def ex4(x_init):
    
    # Function
    def f4(x):
        m1 = [0.0, 0.0]
        m2 = [0.0, 5.0]
        r1 = 1
        r2 = 2
        return np.array([(x[0] - m1[0])**2 + (x[1] - m1[1])**2 - r1**2,
                         (x[0] - m2[0])**2 + (x[1] - m2[1])**2 - r2**2])

    # Jacobian of function
    def J4(x):
        return np.array([[2*x[0], 2*x[1]],
                         [2*x[0], 2*x[1] - 10]])

    # Initial guess
    #x_init = [1.0, 4.0] # converges

    # MYSOLVER
    x, k_end = newton_method1(f4, J4, x_init)
    print(k_end)

    # FSOLVER
    sol = fsolve(f4, x_init)  # Löse nichtlineares Gleichungssystem    
    

    print("Example 4: Two non intersecting circles")
    if x is not None:
        print("My Newton Method Solution:\n", x)
        print("fsolve:\n", sol)
        print("Mysolution = fsolve ? ", np.allclose(sol, x))
        print("Number of Iterations:", k_end)
    else:
        print(f"No solution to this LSE for k_max = {k_end}!")


    # Visualise the solution
    x1_vals = np.linspace(-6, 6, 400)
    x2_vals = np.linspace(-3, 8, 400)
    #plot_intersection(f4, J4, x_init, 'Two Circles not intersecting', x1_vals, x2_vals)       
    return



#x_init = [1.0, 0.0] 
#ex4(x_init)

# Initial guess
#x_init = [1.0, 0.0, -1.0]
#x_init = [-1.0, 0.0, 1.0] # am schnellsten!
#x_init = [-1.0, 0.0, -1.0]


# Example 5: Two intersecting lines in the plane
def ex5(x_init):
    
    # Function
    def f5(x):
        return np.array([2*x[0] + 3*x[1] - 5,
                         4*x[0] - x[1] + 2])

    # Jacobian of function
    def J5(x):
        return np.array([[2*x[0], 3*x[1]],
                         [4*x[0], -1*x[1]]])

    # MYSOLVER
    x, k_end = newton_method1(f5, J5, x_init)

    # FSOLVER
    sol = fsolve(f5, x_init)  # Löse nichtlineares Gleichungssystem
    
    # Print statements
    print("Example 5: Two intersecting lines in the plane")
    print("My Newton Method Solution:\n", x)
    print("fsolve:\n", sol)
    print("Mysolution = fsolve ? ", np.allclose(sol, x))
    print("Number of Iterations:", k_end)

    # Visualise the solution
    x1_vals = np.linspace(-5, 5, 400)
    x2_vals = np.linspace(-5, 5, 400)
    plot_intersection(f5, J5, x_init, 'Intersection of Two Lines', x1_vals, x2_vals)       
    return

# Example 6: Three intersecting planes in space
def ex6(x_init):
    
    # Function
    def f6(x):
        return np.array([2*x[0] + 3*x[1] - 4*x[2] - 7,
                         x[0] - 2*x[1] + x[2] + 1,
                         3*x[0] + x[1] - 2*x[2] - 5])

    # Jacobian of function
    def J6(x):
        return np.array([[2*x[0], 3*x[1], -4*x[2]],
                         [1*x[0], -2*x[1], 1*x[2]],
                         [3*x[0], 1*x[1], -2*x[2]]])

    # MYSOLVER
    x, k_end = newton_method1(f6, J6, x_init)

    # FSOLVER
    sol = fsolve(f6, x_init)  # Löse nichtlineares Gleichungssystem
    
    # Print statements
    print("Example 6: Three intersecting planes in space")
    print("My Newton Method Solution:\n", x)
    print("fsolve:\n", sol)
    print("Mysolution = fsolve ? ", np.allclose(sol, x))
    print("Number of Iterations:", k_end)

    return

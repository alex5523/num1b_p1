# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:56:11 2023

@author: alexandrajohansen

"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import qr
from num1a_p1_main import back_sub

# MYSOLVER
def newton_method1(f, J, x0, TOL_res=1e-10, TOL_inc=1e-10, k_max=100):
    x = np.array(x0, dtype=float)
    k = 0   # count iterations

    while k < k_max:
        res = f(x)  # Residual
        
        if np.linalg.norm(res) < TOL_res:
            return x, k
        
        jacobian = J(x)
        
        # Check if the Jacobian is one-dimensional
        if jacobian.ndim == 1:
            inc = -res / jacobian
        else:
            # Check for singular Jacobian matrix
            if np.linalg.det(jacobian) == 0:
                raise ValueError("Singular Jacobian matrix. Newton method may not converge.")
            
            # Solve linear system using QR decomposition and backwards substitution
            Q, R = qr(jacobian)
            inc = back_sub(R, Q.T @ -res)
        
        if np.linalg.norm(inc) < TOL_inc and np.linalg.norm(res) < TOL_res:
            return x + inc, k               # returns updated x value(s)
        
        x += inc
        k += 1
    
    raise ValueError("Newton method did not converge within the maximum number of iterations.")


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

print("EX1: A circle and a hyperbola", x)
print("Newton Method Solution:", x)
print("Number of Iterations:", k_end)

# FSOLVER
sol = fsolve(f1, x_init)  # Löse nichtlineares Gleichungssystem
print("fsolve =", sol)



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

print("\nEx2: Two circles", x)
print("Newton Method Solution:", x)
print("Number of Iterations:", k_end)

# FSOLVER
sol = fsolve(f2, x_init)  # Löse nichtlineares Gleichungssystem
print("fsolve=", sol)


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

print("\nEX3: Three spheres", x)
print("Newton Method Solution:", x)
print("Number of Iterations:", k_end)

# FSOLVER
sol = fsolve(f3, x_init)  # Löse nichtlineares Gleichungssystem
print("fsolve=", sol)







# Plotting function

def plot_intersection_example(f, J, x_init, title, x1_vals, x2_vals):
    plt.close("all")
    plt.style.use('dspstyle-newton.mplstyle')

    x1_vals, x2_vals = np.meshgrid(x1_vals, x2_vals)

    z_vals = f([x1_vals, x2_vals])

    plt.figure(figsize=(8, 8), facecolor=(1.0, 1.0, 1.0))

    # Plot the equations
    plt.contour(x1_vals, x2_vals, z_vals[0], levels=[0], colors='black', linewidths=2)  # Circle 1
    plt.contour(x1_vals, x2_vals, z_vals[1], levels=[0], colors='blue', linewidths=2)   # Circle 2

    # Initial guess
    plt.scatter(x_init[0], x_init[1], color='red', label='Initial Guess', s=50)

    # Solve using Newton's method
    x, k_end = newton_method1(f, J, x_init)
    
    # Plot the last three points before the intersection
    # for i in range(1, 4):
    #     plt.scatter(x_init[0] + x[0]*i, x_init[1] + x[1]*i, color=f'C{i}', s=50)

    # Mark the intersection point
    plt.scatter(x[0], x[1], color='orange', label='Intersection Point', s=100, zorder=5)

    plt.title(title)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # Set grid specifications
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=1)
    
    # Set tick specifications
    plt.tick_params(axis='both', which='both', labelsize=12, color=(0.2, 0.2, 0.2), direction='in', 
                    bottom=True, top=True, left=True, right=True, labelcolor=(0.2, 0.2, 0.2))

    # AXES
    plt.gca().set_facecolor((0.89, 0.93, 0.96))
    plt.gca().title.set_fontsize(14)
    plt.gca().tick_params(axis='both', which='both', labelsize=12)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['top'].set_linewidth(1)
    plt.gca().spines['right'].set_linewidth(1)
    plt.gca().spines['bottom'].set_linewidth(1)
    plt.gca().spines['left'].set_linewidth(1)
    plt.gca().xaxis.label.set_color((0, 0.27, 0.72))
    plt.gca().yaxis.label.set_color((0, 0.27, 0.72))

    plt.legend(loc='upper left')
    plt.axis('equal')
    plt.show()

# Example 1: Hyperbola and circle
x1_vals = np.linspace(-2, 2, 400)
x2_vals = np.linspace(-2, 2, 400)
plot_intersection_example(f1, J1, [1.0, 1.0], 'Intersection of a Circle and a Hyperbola', x1_vals, x2_vals)

# Example 2: 2 circles
x1_vals = np.linspace(-6, 6, 400)
x2_vals = np.linspace(-3, 8, 400)
plot_intersection_example(f2, J2, [1.0, 4.0], 'Intersection of Two Circles', x1_vals, x2_vals)





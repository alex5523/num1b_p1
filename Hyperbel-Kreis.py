# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:51:56 2020

@author: Estevez

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import qr, solve

# MYSOLVER

def newton_method(f, J, x0, tol_residual=1e-6, tol_increment=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)
    
    for iteration in range(max_iter):
        residual = f(x)
        
        if np.linalg.norm(residual) < tol_residual:
            return x, iteration
        
        jacobian = J(x)
        
        # Solve linear system using QR decomposition
        Q, R = qr(jacobian)
        increment = solve(R, np.dot(Q.T, -residual))
        
        if np.linalg.norm(increment) < tol_increment:
            return x + increment, iteration
        
        x += increment
    
    raise ValueError("Newton method did not converge within the maximum number of iterations.")

#############################
# Example functions
#############################

def funcHypKreis(x):
    # Variablen
    x1, x2 = x
    
    # Gleichungen 
    f1 = x1**2 + x2**2 - 2.0
    f2 = x1**2 - x2**2 - 1.0
    
    return [f1, f2]

# Example usage:
# Define the function and its Jacobian for the circle and hyperbola system
def f(x):
    return np.array([x[0]**2 + x[1]**2 - 2,
                     x[0]**2 - x[1]**2 - 1])

def J(x):
    return np.array([[2*x[0], 2*x[1]],
                     [2*x[0], -2*x[1]]])

##############################
#Execution
##############################
# Initial guess
x_init = [1.0, 1.0]

# Call the Newton method
x, k_end = newton_method(f, J, x_init)

print("Newton Method Solution:", x)
print("Number of Iterations:", k_end)



# FSOLVER
x0 = [1.0, 1.0]  # Startwert für Iteration
sol = fsolve(funcHypKreis, x0)  # Löse nichtlineares Gleichungssystem
print("sol=", sol)


##########################
# Plotting
#########################

# Plot the equations
x1_vals = np.linspace(-2, 2, 400)
x2_vals_f1 = np.sqrt(2 - x1_vals**2)
x2_vals_f2 = np.sqrt(x1_vals**2 - 1)
x2_vals_f1_neg = -np.sqrt(2 - x1_vals**2)
x2_vals_f2_neg = -np.sqrt(x1_vals**2 - 1)

plt.figure(figsize=(8, 6))

# Plot the equations
plt.plot(x1_vals, x2_vals_f1, label=r'§x1^2 + x2^2 - 2.0 = 0§', color='firebrick')
plt.plot(x1_vals, x2_vals_f2, label='x1^2 - x2^2 - 1.0 = 0', color='goldenrod')
plt.plot(x1_vals, x2_vals_f1_neg, color='firebrick')
plt.plot(x1_vals, x2_vals_f2_neg, color='goldenrod')
plt.scatter(sol[0], sol[1], color='blue', label='Solution', s=100, zorder=5, linewidths=1,)

#r'$sig1=5cos(2\pi f_a t) +sin(2\pi x 10f_a t), f_a=1, kHz$'
plt.title('Intersection of a circle and a hyperbola')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend(loc='lower right')
plt.show()

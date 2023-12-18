# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:56:11 2023

@author: alexandrajohansen

"""
import numpy as np
import sys
from scipy.optimize import fsolve
from scipy.linalg import qr
from num1a_p1_main import back_sub

def newton_method1(f, J, x_init, tolerance=1e-10, k_max=100):
    # Parameters/start initializations
    TOL_res = tolerance
    TOL_inc = tolerance
    x = np.array(x_init, dtype=float)
    k = 0   # count iterations

    # Dimension checks
    if np.ndim(x_init) == 1:
        n_unknowns = len(x_init)
    else:
        raise ValueError("Initial guess must be a 1D array (vector).")

    if len(f(x)) != n_unknowns:
        raise ValueError("The number of equations must be equal to the number of unknowns.")

    while k < k_max and np.all(x < 1e6):
        res = f(x)  # Residual
        jacobian = J(x)

        # Check if the Jacobian is one-dimensional
        if jacobian.ndim == 1:
            inc = -res / jacobian
        else:
            # Check jacobian singular
            check_jacobian_singular(jacobian)
            inc = solve_linear_system(jacobian, res)

        # Check using residual if diverging
        divergence_check(f, x, k, k_max, res)

        if no_solution_check(f, x):
            return x, k

        # Update x within the loop
        x += inc
        k += 1

        if convergence_check(inc, res, TOL_inc, TOL_res):
            return x, k

    # If the loop completes without convergence, return None
    return None, k_max



# Solver
def solve_linear_system(jacobian, res):
    Q, R = qr(jacobian)
    return back_sub(R, Q.T @ -res)

# Singularity
def check_jacobian_singular(jacobian):
    condition_number = np.linalg.cond(jacobian, p=np.inf)
    if condition_number > 1 / sys.float_info.epsilon:
        raise ValueError("Singular Jacobian matrix. Newton method may not converge.")

# No solution/infite solutions
def no_solution_check(f, x):
    if np.allclose(f(x), 0):
        return True
    return False

# Divergence
def divergence_check(f, x, k, k_max, res):
    if x is not None and k >= 20 and np.linalg.norm(res[-2]) < np.linalg.norm(res[-1]) < np.linalg.norm(res[0]):
        sol = fsolve(f, x[0, :])
        raise ValueError(f"Newton method is diverging, breaking off at k = {k + 1} with k_max = {k_max}, Estimated solution by fsolve:", sol)

# Comvergence 
def convergence_check(inc, res, TOL_inc, TOL_res):
    if np.linalg.norm(inc) < TOL_inc and np.linalg.norm(res) < TOL_res:
        return True
    return False
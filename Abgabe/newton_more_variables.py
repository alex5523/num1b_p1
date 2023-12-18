# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:56:11 2023

@author: alexandrajohansen

"""
import numpy as np
import sys
from scipy.linalg import qr
from scipy.optimize import fsolve
from num1a_p1_main import back_sub


# MYSOLVER
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
            # Check for singular Jacobian matrix
            condition_number = np.linalg.cond(jacobian, p=np.inf)
            if condition_number > 1/sys.float_info.epsilon:
                raise ValueError("Singular Jacobian matrix. Newton method may not converge.")
            
            # Solve linear system using QR decomposition and backwards substitution
            Q, R = qr(jacobian)
            inc = back_sub(R, Q.T @ -res)
        
        # STOP Divergence check
        if x is not None and k >= 20 and np.linalg.norm(res[-2]) < np.linalg.norm(res[-1]) < np.linalg.norm(res[0]):
            sol = fsolve(f, x[0, :])
            raise ValueError(f"Newton method is diverging, breaking off at k = {k+1} with k_max = {k_max}, Estimated solution by fsolve:", sol)
            
        # STOP No solution or infinite solutions check
        elif np.allclose(f(x), 0):
            return x, k  # returns updated x value(s)
        
        # Update x within the loop
        x += inc
        k += 1
    
    # STOP Convergence check after the loop
    if np.linalg.norm(inc) < TOL_inc and np.linalg.norm(res) < TOL_res:
        return x, k, k_max  # returns updated x value(s)
    
    # If the loop completes without convergence, return None
    return None, k, k_max




"""


# MYSOLVER
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
            # Check for singular Jacobian matrix
            condition_number = np.linalg.cond(jacobian, p=np.inf)
            if condition_number > 1/sys.float_info.epsilon:
                raise ValueError("Singular Jacobian matrix. Newton method may not converge.")
            
            # Solve linear system using QR decomposition and backwards substitution
            Q, R = qr(jacobian)
            inc = back_sub(R, Q.T @ -res)
        
        # STOP Divergence check
        if x is not None and k >= 20 and np.linalg.norm(res[-2]) < np.linalg.norm(res[-1]) < np.linalg.norm(res[0]):
            sol = fsolve(f, x[0, :])
            raise ValueError(f"Newton method is diverging, breaking off at k = {k+1} with k_max = {k_max}, Estimated solution by fsolve:", sol)

        # STOP No solution or infinite solutions check
        elif np.allclose(f(x), 0):
            return x, k, k_max  # returns updated x value(s)

        # Update x within the loop
        x += inc
        k += 1

    # STOP Convergence check after the loop
    if np.linalg.norm(inc) < TOL_inc and np.linalg.norm(res) < TOL_res:
        return x, k, k_max  # returns updated x value(s)

    # If the loop completes without convergence, return None
    return None, k, k_max    
 






        # STOP Divergence check
        if x is not None and k >= 10 and np.linalg.norm(res[-2]) < np.linalg.norm(res[-1]) < np.linalg.norm(res[0]):
            sol = fsolve(f, x)  # this might cause RuntimeError
            #return x, k, k_max
            raise ValueError(f"Newton method is diverging, breaking off at k = {k+1} with k_max = {k_max}, Estimated solution by fsolve:", sol)
        
        # Not working properly: 
        # STOP No solution or infinite solutions check
        elif k > 2 and np.linalg.norm(inc[-2]) and np.linalg.norm(inc[-1]) > TOL_inc:
            return None, k, k_max
            #raise ValueError(f"Newton method did not converge within {k_max} iterations, k_iterations: {k}.")
        


    # If method runs till the end
    #raise ValueError(f"Newton method did not converge within {k_max} iterations.")
    #return None, k_max 

            # Catch RuntimeWarning during fsolve
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                try:
                    sol = fsolve(f, x)
                except RuntimeWarning as e:
                    print(f"fsolve raised a RuntimeWarning: {e}")
                    # Handle the warning as needed
    
            raise ValueError(f"Newton method is diverging, breaking off at k = {k+1} with k_max = {k_max}, Estimated solution by fsolve:", sol)



"""
        
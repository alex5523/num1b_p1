# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:56:11 2023

@author: alexandrajohansen

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import qr
from num1a_p1_main import back_sub

# MYSOLVER
def newton_method1(f, J, x_init, tolerance = 1e-10, k_max=100):
    
    # Parameters/start initialisations
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
                print(k)
                raise ValueError("Singular Jacobian matrix. Newton method may not converge.")
                
            
            # Solve linear system using QR decomposition and backwards substitution
            Q, R = qr(jacobian)
            inc = back_sub(R, Q.T @ -res)
        
        if np.linalg.norm(inc) < TOL_inc and np.linalg.norm(res) < TOL_res:
            return x + inc, k               # returns updated x value(s)
        
        x += inc
        k += 1
    
    else:
        raise ValueError(f"Newton method did not converge for k_max = {k_max}.")





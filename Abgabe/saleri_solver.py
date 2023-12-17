# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:56:11 2023

@author: alexandrajohansen

"""
import numpy as np
from scipy.optimize import fsolve
from newton_more_variables import newton_method1


def newtonsys(F_expr, J_expr, x0, tol, nmax, p):
    n_eq = len(F_expr)
    n_unknowns = len(x0)
    
    if n_eq != n_unknowns:
        raise ValueError('The number of equations must be equal to the number of unknowns for the solver.')

    n = n_eq
    iter_count = 0
    Fxn = np.zeros(n)
    x = x0
    Jxn = np.zeros((n, n))
    err = tol + 1

    def F_func(x):
        variables = {f'x{i+1}': x[i] for i in range(n)}
        return np.array([eval(expr, variables) for expr in F_expr])

    def J_func(x):
        variables = {f'x{i+1}': x[i] for i in range(n)}
        return np.array([[eval(expr, variables) for expr in row] for row in J_expr])

    # Helper functions for LU decomposition
    def lu(A):
        # LU decomposition with partial pivoting
        n = len(A)
        P = np.eye(n)
        L = np.zeros((n, n))
        U = np.copy(A)

        for i in range(n):
            pivot_index = np.argmax(np.abs(U[i:, i])) + i

            if i != pivot_index:
                # Swap rows in U and P
                U[[i, pivot_index], :] = U[[pivot_index, i], :]
                P[[i, pivot_index], :] = P[[pivot_index, i], :]
                if i > 0:
                    L[[i, pivot_index], :i] = L[[pivot_index, i], :i]

            L[i:, i] = U[i:, i] / U[i, i]
            U[i+1:, i:] -= L[i+1:, i:i+1] @ U[i:i+1, i:]

        return L, U, P

    def forward_col(L, b):
        # Forward substitution for a column vector
        n = len(b)
        y = np.zeros_like(b)

        for i in range(n):
            y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]

        return y

    def backward_col(U, y):
        # Backward substitution for a column vector
        n = len(y)
        x = np.zeros_like(y)

        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

        return x

    while err > tol:
        if iter_count % p == 0:
            Fxn = F_func(x)
            Jxn = J_func(x)

            try:
                # LU decomposition
                L, U, P = lu(Jxn)
            except Exception as e:
                raise ValueError(f'LU decomposition failed: {e}')

        else:
            Fxn = F_func(x)

        iter_count += 1

        # Solve linear system
        Fxn = -P @ Fxn
        y = forward_col(L, Fxn)
        deltax = backward_col(U, y)
        x = x + deltax
        err = np.linalg.norm(deltax)

        if iter_count > nmax:
            raise ValueError('Fails to converge within the maximum number of iterations')

    return x, iter_count

# Example usage:
# Saleri et al.
# Define F and J as strings
F_expr = ['x1**2 + x2**2 - 1', 'x1 - x2']
J_expr = [['2*x1', '2*x2'], ['1', '-1']]

# Initial guess
x0 = np.array([0.5, 0.5])

# Tolerance and maximum number of iterations
tolerance = 1e-6
max_iterations = 100

# Number of consecutive steps during which the Jacobian is maintained fixed
p_value = 3

result, iterations = newtonsys(F_expr, J_expr, x0, tolerance, max_iterations, p_value)

print('Result.saleri:', result)
print('Number of iterations:', iterations)


# fsolve
# Define the system of equations
def equations(x):
    x1, x2 = x[0], x[1]
    eq1 = x1**2 + x2**2 - 1
    eq2 = x1 - x2
    return [eq1, eq2]

# Initial guess
x0 = np.array([0.5, 0.5])

# Solve the system using fsolve
result = fsolve(equations, x0)
print('Result.fsolve:', result)


#x, k_end = newton_method1(f3, J3, x_init)

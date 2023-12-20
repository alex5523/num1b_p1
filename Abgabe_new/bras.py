#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:35:31 2023

@author: alexandrajohansen
"""


##############################################################################
# Define and evaluate function expression
##############################################################################
"""
F_expr is a list with equations as elements 

n is the number of unknowns

F_func takes the list of equations and evaluates the expression based on input x (an array)
and outputs the result as a vector

"""
def F_func(x):
    variables = {f'x{i+1}': x[i] for i in range(n)}    
    return np.array([eval(expr, variables) for expr in F_expr])


# Test example
F_expr = ['x1**2 + x2**2 - 2', 'x1**2 - x2**2 - 1']
n = 2

x_init = np.array([1.0, 2.0])
F_result = F_func(x_init)

#for i in range(len(F_result)):
#    print(f"f{i+1}(x{i+1}) = {F_result[i]}")



##############################################################################
# Define and valuate jacobian matrix
##############################################################################

def J_func(x):
        variables = {f'x{i+1}': x[i] for i in range(n)}
        return np.array([[eval(expr, variables) for expr in row] for row in J_expr])

J_expr = [['2*x1', '2*x2'], ['x1', 'x2']]

J_result = J_func(x_init)

#print(J_result)




#iter_count = newtonsys(F_expr, J_expr, x_init)
#print(iter_count)






"""



        
        for iter_count in range(-2,0):
            if err[iter_count-1] == err[iter_count]:
                print("Method converged")
        
        
#        if iter_count > 1 and np.allclose(err[-2:], err[0]):
#            print("Method converged")
#            return x, iter_count






        # Divergence
        if deltax > 3 and x[-2] < x[-1] < x[0]:
            print("Method is diverging")
            return x, iter_count


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




def func(f,x):
    
    
    return fx

def diff(f, x):
    
    fx = sym.lambdify(x,f,'numpy')
    
    dfx = sym.diff(f,x)
    
    df = sym.lambdify(x,dfx,'numpy')
        
    return fx, df



x=sym.Symbol('x') 
f=x**2+10*x+21

fx, df = diff(f)
print("df: ", df)
print("fx: ", fx)




diff_f=sym.diff(f,x)
diff_f
 
f_func=sym.lambdify(x,f,'numpy')
diff_f_func=sym.lambdify(x,diff_f,'numpy')
 
def newtonMethod(x0,iterationNumber, f,df):
    x=x0
     
    for i in range(iterationNumber):
         
        x=x-f(x)/df(x)
     
    residual=np.abs(f(x))
    return x,residual
 
solution,residual = newtonMethod(-2,200,f_func,diff_f_func)



# Useful print statements
#for i in range(len(F_result)):
#    print(f"f{i+1}(x{i+1}) = {F_result[i]}")

# prints every row in jacobi matrix:
for i in range(len(J_result)):
    print(f"j{i+1}(x) = {J_result[i]}")


"""

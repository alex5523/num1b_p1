#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 20:13:04 2024

@author: alexandrajohansen
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import warnings
warnings.filterwarnings("ignore", message="Intel MKL WARNING:")



def normalengleichung(xi, bi, weighted = False):
    
    if len(xi) != len(bi):
        raise ValueError("length of value pairs not equal")

    m = len(xi)    

    ones = np.ones(len(xi)).reshape(len(xi), 1)
    
    xi = xi.reshape(len(xi), 1)
    bi = bi.reshape(len(bi), 1)
    
    A = np.hstack((ones, xi))

    if weighted:
        weights = 1 / (bi ** 2)
        W = np.diagflat(weights)
        
        ATA = np.transpose(A) @ W @ A
        ATb = np.transpose(A) @ W @ bi
    else:
        ATA = np.transpose(A) @ A
        ATb = np.transpose(A) @ bi 
    
    try:
        n, m = np.linalg.solve(ATA, ATb)
    
    except np.linalg.LinAlgError:
        print("Singular matrix encountered. Using pseudoinverse instead.")
        ATA_inv = np.linalg.pinv(ATA)
        n, m = ATA_inv @ ATb

    #n, m = np.linalg.solve(ATA, ATb)
    
#    print(np.linalg.norm(A@xi - bi))
    
    x = np.linspace(min(xi) - 10 ,max(xi) + 10, 10)
    
    y = n + m * x
    
    return n, m, x, y, A

def olsGif(xi, bi, fps, save_path="fitted_line.gif"):
    plt.figure(figsize=(6, 6))
    frames = []

    for i in range(1, len(xi) + 1):  
        plt.clf()

        n, m, x, y, _ = normalengleichung(xi[:i], bi[:i])
        plt.grid(True)
        plt.axis('equal')

        plt.scatter(xi[:i], bi[:i])
        if i > 1:
            plt.plot(x, y, color="g")

        plt.xlim(-10, 30)
        plt.ylim(-10, 10)
        
        plt.legend()
        plt.title(f'm points = {i}')

        fig = plt.gcf()
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())

        frames.append(frame.copy()) 

    imageio.mimsave(save_path, frames, fps=fps)
    plt.close()
    print(f"GIF saved to {save_path}")
    

xi = np.array([1, 4, 9]).reshape(3, 1)
bi = np.array([1, 2, 3]).reshape(3, 1)

olsGif(xi, bi, 0.5, "fitted_line1.gif")

n, m, x, y, A = normalengleichung(xi, bi, True)

print("cond(A) = ", np.linalg.cond(A))



print("y = mx + n")
#print("A:\n", A)
#print("ATA:\n", ATA)
#print("ATb:\n", ATb) 
print("slope m:\n", m)
print("intercept n:\n", n)

del xi, bi, n, m, x, y



xi = np.array([1, 4, 9, 15, 18])
bi = np.array([1, 6, 2, -1, -5])

olsGif(xi, bi, 0.5, "fitted_line2.gif")








#plt.grid(True)
#plt.axis('equal')
#plt.xlim(-2,12)
#plt.ylim(-2,5)
#plt.plot(x, y, color = "g", label = "Fitted line")
#plt.scatter(xi, bi)
#plt.legend()
#plt.show()
#plt.close()

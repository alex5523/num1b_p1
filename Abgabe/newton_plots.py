#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 12:26:50 2023

@author: alexandrajohansen
"""
import numpy as np
import matplotlib.pyplot as plt
from newton_more_variables import newton_method1

# Plotting function
def plot_intersection(f, J, x_init, title, x1_vals, x2_vals):
    plt.close("all")
    
    # Apply custom Matplotlib style
    plt.style.use('dspstyle-newton.mplstyle')

    x1_vals, x2_vals = np.meshgrid(x1_vals, x2_vals)

    z_vals = f([x1_vals, x2_vals])

    # Reduce the figure size
    plt.figure(figsize=(1.8, 1.8), facecolor=(1.0, 1.0, 1.0))

    # Plot the equations
    plt.contour(x1_vals, x2_vals, z_vals[0], levels=[0], colors='black', linewidths=0.6)  # Circle 1
    plt.contour(x1_vals, x2_vals, z_vals[1], levels=[0], colors='blue', linewidths=0.6)   # Circle 2

    # Initial guess
    plt.scatter(x_init[0], x_init[1], color='red', label='Initial Guess', s=3)

    # Solve using Newton's method
    x, k, k_end = newton_method1(f, J, x_init)
    
    # Plot the last three points before the intersection
    # for i in range(1, 4):
    #     plt.scatter(x_init[0] + x[0]*i, x_init[1] + x[1]*i, color=f'C{i}', s=30)

    # Mark the intersection point
    plt.scatter(x[0], x[1], color='orange', label='Intersection Point', s=5, zorder=5)

    plt.title(title, fontsize=6)  # Adjust title fontsize
    plt.xlabel(r'$x_1$', fontsize=4)  # Adjust x-axis label 
    plt.ylabel(r'$x_2$', fontsize=4)  # Adjust y-axis label fontsize
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # Set grid specifications
    plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
    
    # Set tick specifications
    plt.tick_params(axis='both', which='both', labelsize=3, color=(0.2, 0.2, 0.2), direction='in', 
                    bottom=True, top=True, left=True, right=True, labelcolor=(0.2, 0.2, 0.2))

    # AXES
    plt.gca().set_facecolor((0.89, 0.93, 0.96))
    plt.gca().title.set_fontsize(4)  # Adjust axes title fontsize
    plt.gca().tick_params(axis='both', which='both', labelsize=3)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['top'].set_linewidth(0.6)
    plt.gca().spines['right'].set_linewidth(0.6)
    plt.gca().spines['bottom'].set_linewidth(0.6)
    plt.gca().spines['left'].set_linewidth(0.6)
    plt.gca().xaxis.label.set_color((0, 0.27, 0.72))
    plt.gca().yaxis.label.set_color((0, 0.27, 0.72))

    plt.legend(loc='upper left', fontsize=3)  # Adjust legend fontsize
    plt.axis('equal')

    resolution_value = 400
    #plt.savefig("intersection_plot.png", format="png", dpi=resolution_value)
    plt.show()

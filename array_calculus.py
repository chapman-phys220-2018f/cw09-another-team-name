#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###
# Name: Jacob Anabi, Grady Lynch
# Student ID: 2294644, 02297574
# Email: anabi@chapman.edu, grlynch@chapman.edu
# Course: PHYS220/MATH220/CPSC220 Fall 2018
# Assignment: CW09
###

"""
array_calculus.py Module Description:

Computes the derivive of some function f (Df = gradient(x) @ f(x))

Functions:
gradient(x) - computes a gradient matrix based on an array x of domain points
derivative(f, n=100) - computes the derivative of a function func at n domain points

Additonal Notes:
Forward finite difference: (f_i+1 - f_i)/dx
Backward finite difference: (f_i - f_i-1)/dx
Central finite difference: (f_i+1 - f_i-1)/2dx
"""

import numpy as np

def gradient(x):
    """
    gradient(x) function description:

    computes the gradient matrix based on an array x of domain points

    Args:
        x - an array containing the domain points of the function

    Returns:
        2D numpy arry - the gradient matrix, D, which is of the form [[-2, 2, ..., 0][-1, 0, 1, ..., 0], ..., [0, ..., -2, 2]]/(2*dx)
    """
    dx = x[1] - x[0] # dx should be same for each point

    # Creating diagonal arrays to be used in the matrix
    main_diag = np.zeros(len(x)) # main diagonal for the gradient
    pos_diag = np.ones(len(x)-1) # diagonal at index + 1 for the gradient, positive numbers
    neg_diag = np.ones(len(x)-1) # diagonal at index - 1 for the gradient, negative numbers
    main_diag[0], main_diag[-1] = -2, 2 # main diagonal boundarys are -2 and 2 based on forward and backward finite difference
    pos_diag[0] = 2 # positive diagonal front boundary is 2, based on forward finite difference
    neg_diag = neg_diag*-1 # converting each element in the negative diagonal to negative numbers
    neg_diag[-1] = -2 # negative diagonal back boundary is -2, based on backward finite difference

    D = np.diag(main_diag, k=0) + np.diag(pos_diag, k=1) + np.diag(neg_diag, k=-1) # our gradient matrix, D, which is of the form [[-2, 2, ..., 0][-1, 0, 1, ..., 0], ..., [0, ..., -2, 2]]
    return D/(2*dx) # multipling our matrix by the scalar 1/(2*dx)

def derivative(func, n=100):
    """
    derivative(func, n=100) function description:

    Args:
        func - the function to compute the derivative of
        n - the number of domain points (defaults to 100)

    Returns:
        1D numpy array - an array of the function func evaluted at n domain points
    """
    x = np.linspace(0, n-1, n, endpoint=True)
    f = np.vectorize(func)
    Df = gradient(x) @ f(x)
    return Df
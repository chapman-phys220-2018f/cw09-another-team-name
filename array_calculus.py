#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###
# Name: Jacob Anabi, Grady Lynch
# Student ID: 2294644, 02297574
# Email: anabi@chapman.edu, grlynch@chapman.edu
# Course: PHYS220/MATH220/CPSC220 Fall 2018
# Assignment: CW09
###

import numpy as np

def gradient(x):
    dx = x[1] - x[0] # dx should be same for each point

    # Creating diagonal arrays to be used in the matrix
    main_diag = np.zeros(len(x))
    pos_diag = np.ones(len(x)-1)
    neg_diag = np.ones(len(x)-1)
    main_diag[0], main_diag[-1] = -2, 2
    pos_diag[0] = 2
    neg_diag = neg_diag*-1
    neg_diag[-1] = -2

    D = np.diag(main_diag, k=0) + np.diag(pos_diag, k=1) + np.diag(neg_diag, k=-1)
    return D/(2*dx)

def derivative(func, n=100):
    x = np.linspace(0, n-1, n, endpoint=True)
    f = np.vectorize(func)
    Df = gradient(x) @ f(x)
    return Df

f = lambda x : x**2
print(derivative(f, n=10))
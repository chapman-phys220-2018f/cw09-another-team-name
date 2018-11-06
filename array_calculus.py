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
    D = np.ones((len(x), len(x))) # creating a 2-D numpy array of ones with size n x n, where n = len(x)
    dx = x[1] - x[0] # dx should be same for each point
    return D

def derivative(func, n=100):
    x = np.linspace(0, n-1, n, endpoint=True)
    f = np.vectorize(func)
    Df = gradient(x) @ f(x)
    return Df

f = lambda x : x**2
print(derivative(f, n=3))
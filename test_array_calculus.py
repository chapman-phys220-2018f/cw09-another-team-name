"""
test_array_calculus.py Test Module Description:

Checks if the derivative function in array_calculus.py is correct

Functions:
test_derivative - tests the derivative function of array_calculus.py
"""

import math
import array_calculus as ac

def test_derivative():
    """
    test_derivative function description:

    Tests whether the central finite difference is close to the derivative of the function
    """
    f = lambda x: x**2
    Df = ac.derivative(f)
    for i in range (1, len(Df)-1):
        assert math.isclose(Df[i], 2*i)
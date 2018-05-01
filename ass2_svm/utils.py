"""
author: Haicang

Some utility functions for svm.py, but not for algorithms
"""

# If a float x is |x| < delta, I regard it's 0
# using for decrease support vecs
DELTA = 1e-5


# Use to raise Exception
class NotOptimal(ValueError):
    def __init__(self, message='Quadratic Optimazation Fail'):
        self.message = message

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
# %pip install cvxpy
# %pip install scipy
import cvxpy as cp
import scipy as sp
import time

def error(a, b, begin, end,*args):
    if begin > end:
        begin, end = end, begin
    option = ''
    if args:
        option = args[0]
    if option == 'area':
        return abs((2*end + 2)*np.log(end)  \
                - a * end**2              \
                + (-2*b -2) * end         \
                + (-2*begin-2)*np.log(begin)    \
                + a * begin**2              \
                + (2*b + 2) * begin)
    else:
        c = 2.0/(1.0 + np.sqrt(1.0 - 4.0 * a)) 

    return abs(np.log(c) + 1/c - ( a*c +b))

def f(x):
    return 1/x + np.log(x)

def search(begin, end, precision):
    pivot = begin
    while abs(begin - end) > 1e-9:
        mid = (begin + end)/2
        a = (f(mid) - f(pivot))/(mid-pivot)
        b = f(mid) - a*mid
        if error(a,b,pivot,mid) <= precision:
            begin = mid
        else:
            end = mid
    return (begin + end)/2

def approximate(precision,eps):
    lines = []
    curr = 2
    while curr > eps:
        next = search(curr,0,precision)
        a = np.array([    [next  , 1],
                          [curr, 1]   ])
        b = np.array([    f(next),
                          f(curr)        ])
        c = np.linalg.solve(a, b)
        lines.append(c)
        curr = next
    return np.array(lines)
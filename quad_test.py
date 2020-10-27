# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1x4jioIXp9LqsC8Uep_-5wKYRC9arkrr0
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
# %pip install cvxpy
# %pip install scipy
import cvxpy as cp
import scipy as sp
import time
import ps
import os

np.set_printoptions(precision = 3)

def f(x):
    return np.log(x) + 1/x

def lagrange_interpolate(xa,xb,xc):
    M = np.array([[xa**2, xa, 1],
                  [xb**2, xb, 1],
                  [xc**2, xc, 1]])
    B = np.array([f(xa),f(xb),f(xc)])
    A = np.linalg.solve(M, B)
    return A

def error(xa,xb,xc):
    A = lagrange_interpolate(xa,xb,xc)
    X = np.roots([-2*A[0],-A[1],1,-1])
    X = X[X>0]
    error = abs(f(X) - (A[0]*X**2 + A[1]*X + A[2])) 
    return max(error)


def search2(xa,xc):
    begin = xa
    end = xc
    while abs(begin - end) > 1e-9:
        xb = (begin + end)/2
        if error(xa,xb,xc) <= precision:
            begin = xb
        else:
            end = xb
    xb = (begin + end)/2 
    return error(xa,xb,xc),xb


def search(begin, end, precision):
    pivot = begin
    while abs(begin - end) > 1e-9:
        mid = (begin + end)/2
        a = (f(mid) - f(pivot))/(mid-pivot)
        b = f(mid) - a*mid
        if search2(pivot,mid)[0] <= precision:
            begin = mid
        else:
            end = mid
    return (begin + end)/2

def approximate(precision,eps):
    lines = []
    curr = 2
    while curr - eps > 1e-8:
        next = search(curr,eps,precision)
        c = [curr, search2(curr, next)[1], next]
        #print(c)
        lines.append(c)
        curr = next
    return np.array(lines)


precision = 1e-3
eps = 5e-3
'''
file = 'cubic - ' + str(precision) + '-' + str(eps) + '.npy'
if os.path.isfile(os.path.join('preset', file)):
    coeff  = np.load(os.path.join('preset', file))
else:
    curves = approximate(precision,eps)
    coeff = np.zeros((len(curves),4))
    for i in range(len(curves)):
        coeff[i,:] = lagrange_interpolate(curves[i][0],curves[i][1],curves[i][2],curves[i][3])
    np.save(os.path.join('preset', file), coeff)
'''

curves = approximate(precision,eps)
coeff = np.zeros((len(curves),3))
for i in range(len(curves)):
    coeff[i,:] = lagrange_interpolate(curves[i][0],curves[i][1],curves[i][2])
    #print('{}x^2 + {}x + {}'.format(coeff[i,0],coeff[i,1],coeff[i,2]))

#print(coeff)
print(len(coeff))





np.set_printoptions(precision = 3)
d = 4
n = 50
start_total = time.time()
mean = np.zeros(d)
covariance = np.zeros((d,d))

start = time.time()
print("Generating covariance matrix")
count = 0
while True:
    count = count + 1
    for i in range(d):
        for j in range(d):
            if i == j:
                covariance[i,j] = 1.0
            elif i < j:
                covariance[i,j] = np.random.randint(1,50)/50.0
            else:
                covariance[i,j] = covariance[j,i]
    if ps.is_pos_def(covariance):
        print("Try #",count + 1)
        print(covariance)
        break
    else:
        continue


end = time.time()
print("Generated in {} second\n".format(end - start))
data = np.random.multivariate_normal(mean, covariance, n)


precision = 1e-3
eps = 5e-3
start = time.time()
print("Calculating SN")
sigma = ps.Sn(data)
end = time.time()
print("Calculated in {} second\n".format(end - start))
sqrt_sigma = sp.linalg.sqrtm(sigma)

Z1 = cp.Variable((d,d))
Z2 = cp.Variable((d,d))
Y1 = cp.Variable((d,d))
Y2 = cp.Variable((d,d))
X1 = cp.Variable((d,d), PSD = True)
X2 = cp.Variable((d,d), PSD = True)
X = cp.Variable((d,d), PSD = True)
Y = cp.Variable((d,d))
Z = cp.Variable((d,d))
t = cp.Variable()
I = np.eye(d)

start = time.time()
print("Generating representation")

constraints =  [cp.bmat([[I,X1 * np.sqrt(coeff[i][0])],[X1.T * np.sqrt(coeff[i][0]),Y1 - Z1 * coeff[i][2] - coeff[i][1] * X1]]) >> cp.bmat([[0*I,0*I],[0*I,0*I]]) for i in range(2) ]
constraints += [cp.bmat([[I,X2 * np.sqrt(coeff[i][0])],[X2.T * np.sqrt(coeff[i][0]),Y2 - Z2 * coeff[i][2] - coeff[i][1] * X2]]) >> cp.bmat([[0*I,0*I],[0*I,0*I]]) for i in range(2) ]
constraints += [Z1 + Z2 == I]
constraints += [Z1 >> I*0]
constraints += [Z2 >> I*0]
constraints += [X1 + X2 == X]
constraints += [Y1 + Y2 == Y]
constraints += [X1 << 2*I]
constraints += [X1 >> eps*I]
constraints += [X2 << 2*I]
constraints += [X2 >> eps*I]
constraints += [cp.trace(Y) <= t]
constraints += [X << 2*I]
constraints += [X >> eps*I]
constraints += [(sqrt_sigma @ X @ sqrt_sigma)[i,i] == 1 for i in range(d) ]
constraints += [(sqrt_sigma @ X @ sqrt_sigma) == Z ]


end = time.time()
print("Generated in {} second\n".format(end - start))

print("Solving")
prob = cp.Problem(cp.Minimize(t),constraints)
start = time.time()
prob.solve(solver = cp.MOSEK)
end_total = time.time()

# Print result.
print("Solved in {} second\n".format(end_total - start))
print("The optimal value is", prob.value)
print("Z = \n",X.value)
print("eig = \n",np.linalg.eig(X.value)[0])
print("X = \n",Z.value)

print("Y = \n",Y.value)
print("t = \n",t.value)

print("error = \n", sqrt_sigma @ np.array(Z.value) @ sqrt_sigma - covariance)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
# %pip install cvxpy
# %pip install scipy
import cvxpy as cp
import scipy as sp
import time
import os 

def f(x):
    return 1/x + np.log(x)

def cubicLagrangeInterpolate(xa,xb,xc,xd):
    M = np.array([[xa**3, xa**2, xa, 1],
                  [xb**3, xb**2, xb, 1],
                  [xc**3, xc**2, xc, 1],
                  [xd**3, xd**2, xd, 1]])
    B = np.array([f(xa),f(xb),f(xc),f(xd)])
    A = np.linalg.solve(M, B)
    return A

def linearEstimatorError(begin, end):
    if begin > end:
        begin, end = end, begin
    a = (f(end) - f(begin))/(end-begin)
    b = f(end) - a*end
    c = 2.0/(1.0 + np.sqrt(1.0 - 4.0 * a)) 
    return abs(np.log(c) + 1/c - ( a*c +b))

def cubicEstimatorError(begin,end):
    if begin < end:
        xa = begin
        xd = end
    else:
        xa = end
        xd = begin
    xb = xa + 1/4*(xd-xa)
    xc = xa + 3/4*(xd-xa)
    A = cubicLagrangeInterpolate(xa,xb,xc,xd)
    X = np.roots([-3*A[0],-2*A[1],-A[2],1,-1])
    X = X[X>0]
    error = abs(f(X) - (A[0]*X**3 + A[1]*X**2 + A[2]*X + A[3])) 
    return max(error)

def search(begin, end, precision, estimationType = 'Linear'):
    pivot = begin
    while abs(begin - end) > 1e-9:
        mid = (begin + end)/2

        if estimationType == 'Cubic':
            error = cubicEstimatorError(pivot,mid)
        else:
            error = linearEstimatorError(pivot,mid)

        if error <= precision:
            begin = mid
        else:
            end = mid
    return (begin + end)/2

def approximate(precision,eps,estimationType = 'Linear'):
    file = 'Type' + estimationType + '-Precision' + str(precision) + '-Epsilon' + str(eps) + '.npy'
    if os.path.isfile(os.path.join('preset', file)):
        lines  = np.load(os.path.join('preset', file))
    else:

        lines = []
        curr = 2
        while curr - eps > 1e-9:
            next = search(curr,eps,precision,estimationType)
            if estimationType == 'Cubic':
                c = [curr, next+3/4*(curr-next), next+1/4*(curr-next), next]
            else:
                a = np.array([    [next  , 1],
                                  [curr, 1]   ])
                b = np.array([    f(next),
                                  f(curr)        ])
                c = np.linalg.solve(a, b)
            lines.append(c)
            curr = next
        np.save(os.path.join('preset', file), lines)
    return np.array(lines)

def Sn(data):
    d = data[0].shape[0]
    n = len(data)
    Sn = np.zeros((d,d))
    for i in range(n):
        X = np.expand_dims(data[i], axis=0).T
        Sn += 1/n * X @ X.T
    return Sn


    
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

class LGC:
    def __init__(self,X,dimension,precision = 1e-4,epsilon = 5e-3,estimationType = 'linear'):
        self.h = precision
        self.eps = epsilon
        self.dim = dimension
        self.constraints = []
        self.I = np.eye(self.dim)
        self.estimationType = estimationType
        self.size = 0
        self.X = X
        self.Y = cp.Variable((self.dim,self.dim))
        self.t = cp.Variable()
        self.alpha1 = []
        self.alpha2 = []
        self.alpha3 = []
        self.alpha4 = []
        self.X_ = []
        self.Y_ = []
        self.Z_ = []

    def setup_constraint(self):
        lines = approximate(self.h,self.eps,self.estimationType)
        self.size = len(lines)
        if self.estimationType == 'Cubic':
            for i in range(self.size):
                self.alpha1.append(cp.Variable((self.dim,self.dim)))
                self.alpha2.append(cp.Variable((self.dim,self.dim)))
                self.alpha3.append(cp.Variable((self.dim,self.dim)))
                self.alpha4.append(cp.Variable((self.dim,self.dim)))
                self.X_.append(cp.Variable((self.dim,self.dim), PSD = True))
                self.Y_.append(cp.Variable((self.dim,self.dim)))
                self.Z_.append(cp.Variable((self.dim,self.dim)))
            for i in range(self.size):
                Xi = np.flip(lines[i]).tolist()
                Yi = np.flip(f(lines[i])).flatten().tolist()
                self.constraints += [cp.bmat ([ 
                                                [3.0 * self.alpha2[i],   0.0 * self.I        ],
                                                [0.0 * self.I        ,  12.0 * self.alpha4[i]]
                                        ]) + 
                            cp.bmat     ([      [1.0 * self.alpha3[i],   2.0 * self.alpha2[i]],
                                                [2.0 * self.alpha3[i],   4.0 * self.alpha3[i]]
                                    ]) >> 0 * np.eye(2 * self.dim)]

                self.constraints += [cp.bmat ([ 
                                                [3.0 * self.alpha3[i],   0.0 * self.I        ],
                                                [0.0 * self.I        ,  12.0 * self.alpha1[i]]
                                        ]) + \
                            cp.bmat     ([      [1.0 * self.alpha2[i],   2.0 * self.alpha2[i]],
                                                [2.0 * self.alpha2[i],   4.0 * self.alpha3[i]]
                                    ]) >> 0 * np.eye(2 * self.dim)]

                self.constraints += [self.alpha1[i] + self.alpha2[i] + self.alpha3[i] + self.alpha4[i] == self.Z_[i]]
                self.constraints += [self.alpha1[i] * Xi[0]  + self.alpha2[i] * Xi[1]  + self.alpha3[i] * Xi[2]  + self.alpha4[i] * Xi[3]  == self.X_[i]]
                self.constraints += [self.alpha1[i] * Yi[0]  + self.alpha2[i] * Yi[1]  + self.alpha3[i] * Yi[2]  + self.alpha4[i] * Yi[3]  << self.Y_[i]]
                self.constraints += [self.Z_[i] >> 0 * self.I]
                self.constraints += [cp.sum(self.X_) == self.X]
                self.constraints += [cp.sum(self.Y_) == self.Y]
                self.constraints += [cp.sum(self.Z_) == self.I]
        else:
            a = lines[:,0]
            c = lines[:,1]
            self.constraints += [a[i] * self.X - self.Y << -c[i] * self.I for i in range(self.size)]  
        self.constraints += [self.X << 2*self.I]
        self.constraints += [self.X >> self.eps*self.I]
        self.constraints += [cp.trace(self.Y) <= self.t]

        

    def add_constraint(self, constraints):
        self.constraints += constraints

    def solve(self, solver = cp.SCS,verbose = True):
        self.setup_constraint()
        prob = cp.Problem(cp.Minimize(self.t),self.constraints)
        prob.solve(solver = solver, verbose = verbose)
        return self.t.value, self.X.value

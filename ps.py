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
#input double/numpy array
def f(x):
    return 1/x + np.log(x)
#output double/numpy array

#input 2 double
def linearLagrangeInterpolateF(xa,xb):
    M = np.array([[xa, 1],
                  [xb, 1]])
    B = np.array([f(xa),f(xb)])
    A = np.linalg.solve(M, B)
    return A
#output numpy array (1,2)

#input 2 double
def linearLagrangeInterpolateLog(xa,xb):
    M = np.array([[xa, 1],
                  [xb, 1]])
    B = np.array([np.log(xa),np.log(xb)])
    A = np.linalg.solve(M, B)
    return A
#output numpy array (1,2)

#input number
def linearEstimatorError(begin, end):
    #swap end
    if begin > end:
        begin, end = end, begin

    [a,b] = linearLagrangeInterpolateF(begin,end)
    #y = ax + b 
    c = (1.0 - np.sqrt(1.0 - 4.0 * a))/(2*a)
    return abs(np.log(c) + 1/c - ( a*c +b)),c

def altLinEstimatorError(begin, end):
    if begin > end:
        begin, end = end, begin
    a = (np.log(end) - np.log(begin))/(end-begin)
    b = -a*begin + np.log(begin)
    c = 1/a
    return abs(np.log(c) - (a*c + b)),c

def search(begin, end, precision, estimationType = 'Linear', deltaType = 'max'):
    pivot = begin
    while abs(begin - end) > 1e-9:
        mid = (begin + end)/2

        if estimationType == 'AltLin':
            error,xerr = altLinEstimatorError(pivot,mid)
            if deltaType == 'snr':
                error = error/f(xerr)
        else:
            error,xerr = linearEstimatorError(pivot,mid)
            if deltaType == 'snr':
                error = error/f(xerr)

        if error <= precision:
            begin = mid
        else:
            end = mid
    return (begin + end)/2

def approximate(precision,begin,end,estimationType,deltaType = 'max'):
    #print(estimationType)
    waypoint = []
    lines = []
    curr = end
    waypoint.append(curr)
    while curr > begin + 1e-9:
        #print(curr)
        if estimationType == 'AltLin':
            next = search(curr,begin,precision,'AltLin',deltaType)
            c = linearLagrangeInterpolateLog(next,curr) 
        elif estimationType == 'Linear':
            next = search(curr,begin,precision,'Linear',deltaType)
            c = linearLagrangeInterpolateF(next,curr)   
        lines.append(c)
        waypoint.append(next)
        curr = next
    return np.array(lines),np.array(waypoint)

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
    def __init__(self,cov,dimension, data = [],delta = 1e-3,epsilon = 5e-3,mu = 1e-2, errorType = 'max',debug = False):
        self.dlt = delta
        self.eps = epsilon
        self.dim = dimension
        self.errorType = errorType
        self.mu = mu
        self.constraints = []
        self.data = data
        self.I = np.eye(self.dim)
        self.size = 0
        self.cov = cov
        self.X = cp.Variable((self.dim,self.dim), PSD = True)
        self.Y = cp.Variable((self.dim,self.dim))
        self.t = cp.Variable()
        self.lamd_1 = cp.Variable((self.dim,self.dim))
        self.lamd_2 = cp.Variable((self.dim,self.dim))
        self.X_2 = cp.Variable((self.dim,self.dim))
        self.Y_2 = cp.Variable((self.dim,self.dim))
        self.X_1 = cp.Variable((self.dim,self.dim))
        self.Y_1 = cp.Variable((self.dim,self.dim))
        self.X_ = []
        self.lamd = []
        self.tau = []
        self.taup = []
        self.debug = debug

    def setup_constraint(self):
        self.sigma = Sn(self.data)
        self.sqrt_sigma = sp.linalg.sqrtm(self.sigma)

        self.linesFrL,self.waypointFrL = approximate(self.dlt,self.eps,self.mu,'AltLin',self.errorType)
        self.linesLin,self.waypointLin = approximate(self.dlt,self.mu,2,'Linear',self.errorType)
        if (self.debug):
            print('{} alternate segments'.format(len(self.linesFrL)))
            print('{} linear segments'.format(len(self.linesLin)))
        #setup array variables
        for j in range(len(self.linesFrL)):
            self.lamd.append(cp.Variable((self.dim,self.dim), PSD = True))
            self.X_.append(cp.Variable((self.dim,self.dim), PSD = True))
            self.taup.append(cp.Variable((self.dim,self.dim), PSD = True))
            self.tau.append(cp.Variable((self.dim,self.dim), PSD = True))

        #AltLin portion
        for j in range(len(self.linesFrL)):
            a = self.linesFrL[j][0]
            b = self.linesFrL[j][1]
            self.constraints += [cp.bmat ([   [self.X_[j],self.lamd[j]],[self.lamd[j],self.taup[j]]]) >> 0 * np.eye(2 * self.dim)]
            self.constraints += [self.taup[j] + a*self.X_[j] + b*self.lamd[j] << self.tau[j]]

        self.constraints += [cp.sum(self.X_) == self.X_1]
        self.constraints += [cp.sum(self.tau) == self.Y_1]
        self.constraints += [cp.sum(self.lamd) == self.lamd_1]

        #Linear Portion
        a = self.linesLin[:,0]
        c = self.linesLin[:,1]
        self.constraints += [a[i] * self.X_2 - self.Y_2 << -c[i] * self.lamd_2 for i in range(len(self.linesLin))]

        self.constraints += [self.lamd_1 + self.lamd_2 == self.I]
        self.constraints += [self.X_1 + self.X_2 == self.X]
        self.constraints += [self.Y_1 + self.Y_2 == self.Y]


        self.constraints += [self.X << 2*self.I]
        self.constraints += [self.X >> self.eps*self.I]
        self.constraints += [cp.trace(self.Y) <= self.t]

        self.constraints += [self.sqrt_sigma @ self.X @ self.sqrt_sigma == self.cov]

    def add_constraint(self, constraints):
        self.constraints += constraints

    def solve(self, solver = cp.SCS,verbose = True):
        self.setup_constraint()
        prob = cp.Problem(cp.Minimize(self.t),self.constraints)
        prob.solve(solver = solver, verbose = verbose)
        self.time = prob.solver_stats.solve_time
        return self.t.value, self.cov.value, self.time

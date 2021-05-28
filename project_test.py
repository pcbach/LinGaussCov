import cvxpy as cp
import numpy as np
import ps

np.set_printoptions(precision = 2, suppress=True)

d = 10
precision = 5e-4
eps = 1e-5
mu = 1e-3

cov = cp.Variable((d,d))
Problem = ps.LGC(cov,d,data = [],delta = precision, epsilon = eps,mu = mu, error_type = 'snr', debug = True)

print(Problem.solve(cp.MOSEK, verbose = True))
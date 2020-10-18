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

np.set_printoptions(precision = 3)
d = 4
n = 5000
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


precision = 1e-4
eps = 5e-3
Z = cp.Variable((d,d), PSD = True)

sigma = ps.Sn(data)
sqrt_sigma = sp.linalg.sqrtm(sigma)


Problem = ps.LGC(precision,eps,d,Z)

Problem.add_constraint([(sqrt_sigma @ Problem.Z @ sqrt_sigma)[i,i] == 1 for i in range(d) ])

ans = Problem.solve(cp.MOSEK)

z = ans[1]
t = ans[0]

print("Z = \n",z)
print("eig = \n",np.linalg.eig(z)[0])
print("SZS = \n",sqrt_sigma @ np.array(z) @ sqrt_sigma)

print("t = \n",t)

print("error = \n", sqrt_sigma @ np.array(z) @ sqrt_sigma - covariance)



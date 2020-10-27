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

np.set_printoptions(precision = 3, suppress=True)
d = 3
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
                covariance[i,j] = np.random.randint(-50,50)/100.0
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
X = cp.Variable((d,d), PSD = True)

sigma = ps.Sn(data)
sqrt_sigma = sp.linalg.sqrtm(sigma)

start = time.time()
Problem1 = ps.LGC(X,d,precision = 1e-4, epsilon = 5e-4,estimationType = 'Cubic')
Problem1.add_constraint([(sqrt_sigma @ Problem1.X @ sqrt_sigma)[i,i] == 1 for i in range(d) ])
ans1 = Problem1.solve(cp.MOSEK, verbose = True)
z1 = ans1[1]
t1 = ans1[0]
end = time.time()
cubicTime = end-start

start = time.time()
Problem2 = ps.LGC(X,d,precision = 1e-4, epsilon = 5e-4,estimationType = 'Linear')
Problem2.add_constraint([(sqrt_sigma @ Problem2.X @ sqrt_sigma)[i,i] == 1 for i in range(d) ])
ans2 = Problem2.solve(cp.MOSEK, verbose = True)
z2 = ans2[1]
t2 = ans2[0]
end = time.time()
linearTime = end-start

cubicParserError = sqrt_sigma @ np.array(z1) @ sqrt_sigma - covariance
linearParserError = sqrt_sigma @ np.array(z2) @ sqrt_sigma - covariance

start = time.time()
D = np.sqrt(np.diag(sigma))
scale = np.linalg.inv(np.diag(D))
sampleError = scale @ sigma @ scale - covariance
end = time.time()
sampleTime = end-start

print("┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓")
print("┃Estimator┃ Sum Error ┃ Max Error ┃    Time   ┃")
print("┣━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━┫")
print("┃  Cubic  ┃  {:7.3f}  ┃  {:7.3f}  ┃  {:7.3f}  ┃".format( float(np.sum(abs(cubicParserError))),np.max(abs(cubicParserError)),cubicTime))
print("┣━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━┫")
print("┃  Linear ┃  {:7.3f}  ┃  {:7.3f}  ┃  {:7.3f}  ┃".format( float(np.sum(abs(linearParserError))),np.max(abs(linearParserError)),linearTime))
print("┣━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━┫")
print("┃  Sample ┃  {:7.3f}  ┃  {:7.3f}  ┃  {:7.3f}  ┃".format( float(np.sum(abs(sampleError))),np.max(abs(sampleError)),sampleTime))
print("┗━━━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━━━┛")



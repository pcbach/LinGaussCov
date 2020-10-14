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

Z = cp.Variable((d,d), PSD = True)
Y = cp.Variable((d,d))
t = cp.Variable()
I = np.eye(d)

start = time.time()
print("Generating representation")
constraints = ps.represent(d, precision, eps, Z, Y)
constraints += [Z << 2*I]
constraints += [Z >> eps*I]
constraints += [cp.trace(Y) <= t]
constraints += [(sqrt_sigma @ Z @ sqrt_sigma)[i,i] == 1 for i in range(d) ]
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
print("Z = \n",Z.value)
print("eig = \n",np.linalg.eig(Z.value)[0])
print("SZS = \n",sqrt_sigma @ np.array(Z.value) @ sqrt_sigma)

print("Y = \n",Y.value)
print("t = \n",t.value)

print("error = \n", sqrt_sigma @ np.array(Z.value) @ sqrt_sigma - covariance)

print("Total time {} second\n".format(end_total - start_total))

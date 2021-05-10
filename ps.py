import cvxpy
import numpy
import scipy
import mosek


def f(x: float) -> float:
    """
    return value of function 1/x + log(x)
    :param x: x
    :return: f(x)
    """
    return 1 / x + numpy.log(x)


def lineParameterF(xa: float, xb: float) -> numpy.ndarray:
    """
    return value of the line y = ax + b 
    that run through the points (xa,f(xa)), (xb,f(xb))
    :param xa: begin
    :param xb: end
    :return: array [a, b] for line y = ax + b
    """
    m: numpy.ndarray = numpy.array([[xa, 1],
                                    [xb, 1]])
    b: numpy.ndarray = numpy.array([f(xa), f(xb)])
    a: numpy.ndarray = numpy.linalg.solve(m, b)
    return a


def lineParameterLog(xa: float, xb: float) -> numpy.ndarray:
    """
    return value of the line y = ax + b
    that run through the points (xa,ln(xa)), (xb,ln(xb))
    :param xa: begin
    :param xb: end
    :return: array [a, b] for line y = ax + b
    """
    m: numpy.ndarray = numpy.array([[xa, 1],
                                    [xb, 1]])
    b: numpy.ndarray = numpy.array([numpy.log(xa), numpy.log(xb)])
    a: numpy.ndarray = numpy.linalg.solve(m, b)
    return a


def linearEstimatorError(begin: float, end: float) -> (float, float):
    """
    return the maximum absolute different
    between the line that run through the
    points (begin, f(begin)) and (end, f(end))
    and function f in [begin, end]
    :param begin: begin
    :param end: end
    :return: error value and its position
    """
    # swap end
    if begin > end:
        begin, end = end, begin

    # get line parameter
    [a, b] = lineParameterF(begin, end)
    # y = ax + b
    c = (1.0 - numpy.sqrt(1.0 - 4.0 * a)) / (2 * a)
    return abs(numpy.log(c) + 1 / c - (a * c + b)), c


def altLinEstimatorError(begin: float, end: float) -> (float, float):
    """
    return the maximum absolute different
    between the line that run through the
    points (begin, ln(begin)) and (end, ln(end))
    and ln in [begin, end]
    :param begin: begin
    :param end: end
    :return: error value and its position
    """
    # swap end
    if begin > end:
        begin, end = end, begin
    # get line parameter
    a = (numpy.log(end) - numpy.log(begin)) / (end - begin)
    b = -a * begin + numpy.log(begin)
    # y = ax + b
    c = 1 / a
    return abs(numpy.log(c) - (a * c + b)), c


def search(begin: float, end: float, delta: float, estimation_type: str = 'Linear', delta_type: str = 'max') -> float:
    """
    search for point x' that satisfy:
    - x' is in [begin, end]
    - error of line through (x', f(x')) and (begin, f(begin)) <= delta
    - x' is min
    :param begin: begin
    :param end: end
    :param delta: delta
    :param estimation_type: either 'Linear' or 'AltLin'
    :param delta_type: either 'max' or 'snr'
    :return:
    """
    pivot = begin
    while abs(begin - end) > 1e-9:
        mid = (begin + end) / 2

        if estimation_type == 'AltLin':
            [error, x_err] = altLinEstimatorError(pivot, mid)
            if delta_type == 'snr':
                error = error / f(x_err)
        else:
            error, x_err = linearEstimatorError(pivot, mid)
            if delta_type == 'snr':
                error = error / f(x_err)

        if error <= delta:
            begin = mid
        else:
            end = mid
    return (begin + end) / 2


def approximate(delta: float, begin: float, end: float,
                estimation_type: str, delta_type: str = 'max') -> (numpy.ndarray, numpy.ndarray):
    """
    return the function f(x) representation
    :param delta: delta
    :param begin: begin
    :param end: end
    :param estimation_type: either 'Linear' or 'AltLin'
    :param delta_type: either 'max' or 'snr'
    :return: first array contain the line parameter, the second array contain the endpoints of segment
    """
    # print(estimationType)
    waypoint = []
    lines = []
    curr = end
    waypoint.append(curr)
    while curr > begin + 1e-9:
        # print(curr)
        if estimation_type == 'AltLin':
            next = search(curr, begin, delta, 'AltLin', delta_type)
            c = lineParameterLog(next, curr)
            lines.append(c)
            waypoint.append(next)
        elif estimation_type == 'Linear':
            next = search(curr, begin, delta, 'Linear', delta_type)
            c = lineParameterF(next, curr)
            lines.append(c)
            waypoint.append(next)
        curr = next
    return numpy.array(lines), numpy.array(waypoint)


def Sn(data: numpy.ndarray) -> numpy.ndarray:
    """
    return the sample covariance of data
    :param data: numpy array of shape (n, d); n samples, d dimension
    :return: the sample covariance of size (d, d)
    """
    d = data[0].shape[0]
    n = len(data)
    s_n = numpy.zeros((d, d))
    for i in range(n):
        x = numpy.expand_dims(data[i], axis=0).T
        s_n += 1 / n * x @ x.T
    return s_n


def is_pos_def(x: numpy.ndarray) -> bool:
    """
    check if matrix x is positive-semi-definite
    :param x: the numpy array of shape (k, k)
    :return: return true if matrix x is positive-semi-definite
    """
    return numpy.all(numpy.linalg.eigvals(x) > 0)


class LGC:
    def __init__(self, cov: cvxpy.Variable, dimension: int, data: numpy.ndarray,
                 delta: float = 1e-3, epsilon: float = 5e-3, mu: float = 1e-2,
                 error_type: str = 'max',
                 debug: bool = False) -> None:
        """
        initialization of the parser
        :param cov: the cvxpy variable for the covariance
        :param dimension: the dimension d
        :param data: the data of shape (n,d)
        :param delta: maximum error
        :param epsilon: min x coordinate
        :param mu: estimation method split point
        :param error_type: either 'snr' or 'max'
        :param debug: True for debug mode
        """
        self.dlt = delta
        self.eps = epsilon
        self.dim = dimension
        self.errorType = error_type
        self.mu = mu
        self.constraints = []
        self.data = data
        self.identity = numpy.eye(self.dim)
        self.size = 0
        self.cov = cov
        self.X = cvxpy.Variable((self.dim, self.dim), PSD=True)
        self.Y = cvxpy.Variable((self.dim, self.dim))
        self.t = cvxpy.Variable()
        self.lambda_1 = cvxpy.Variable((self.dim, self.dim))
        self.lambda_2 = cvxpy.Variable((self.dim, self.dim))
        self.X_2 = cvxpy.Variable((self.dim, self.dim))
        self.Y_2 = cvxpy.Variable((self.dim, self.dim))
        self.X_1 = cvxpy.Variable((self.dim, self.dim))
        self.Y_1 = cvxpy.Variable((self.dim, self.dim))
        self.X_ = []
        self.lambda_total = []
        self.tau = []
        self.tau_p = []
        self.debug = debug
        self.prob = None

    def setup_constraint(self):
        sigma = Sn(self.data)
        sqrt_sigma = scipy.linalg.sqrtm(sigma)

        lines_alt_lin, waypoint_alt_lin = approximate(self.dlt, self.eps, self.mu, 'AltLin', self.errorType)
        lines_lin, waypoint_lin = approximate(self.dlt, self.mu, 2, 'Linear', self.errorType)
        if self.debug:
            print('{} alternate segments'.format(len(lines_alt_lin)))
            print('{} linear segments'.format(len(lines_lin)))
        # setup array variables
        for j in range(len(lines_alt_lin)):
            self.lambda_total.append(cvxpy.Variable((self.dim, self.dim), PSD=True))
            self.X_.append(cvxpy.Variable((self.dim, self.dim), PSD=True))
            self.tau_p.append(cvxpy.Variable((self.dim, self.dim), PSD=True))
            self.tau.append(cvxpy.Variable((self.dim, self.dim), PSD=True))

        # AltLin portion
        for j in range(len(lines_alt_lin)):
            a = lines_alt_lin[j][0]
            b = lines_alt_lin[j][1]
            self.constraints += [
                cvxpy.bmat([[self.X_[j], self.lambda_total[j]],
                            [self.lambda_total[j], self.tau_p[j]]])
                >> 0 * numpy.eye(2 * self.dim)]
            self.constraints += [self.tau_p[j] + a * self.X_[j] + b * self.lambda_total[j] << self.tau[j]]

        self.constraints += [cvxpy.sum(self.X_) == self.X_1]
        self.constraints += [cvxpy.sum(self.tau) == self.Y_1]
        self.constraints += [cvxpy.sum(self.lambda_total) == self.lambda_1]

        # Linear Portion
        a = lines_lin[:, 0]
        c = lines_lin[:, 1]
        self.constraints += [a[i] * self.X_2 - self.Y_2 << -c[i] * self.lambda_2 for i in range(len(lines_lin))]

        self.constraints += [self.lambda_1 + self.lambda_2 == self.identity]
        self.constraints += [self.X_1 + self.X_2 == self.X]
        self.constraints += [self.Y_1 + self.Y_2 == self.Y]

        self.constraints += [self.X << 2 * self.identity]
        self.constraints += [self.X >> self.eps * self.identity]
        self.constraints += [cvxpy.trace(self.Y) <= self.t]

        self.constraints += [sqrt_sigma @ self.X @ sqrt_sigma == self.cov]

    def add_constraint(self, constraints):
        self.constraints += constraints

    def solve(self, solver=cvxpy.SCS, verbose=True):
        self.setup_constraint()
        self.prob = cvxpy.Problem(cvxpy.Minimize(self.t), self.constraints)
        self.prob.solve(solver=solver, verbose=verbose)
        time = self.prob.solver_stats.solve_time
        return self.cov.value, time


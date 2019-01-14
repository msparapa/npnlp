from npnlp import minimize
import numpy as np

tol = 1e-6

def test_sqp1():
    def J(x):
        return np.array([x[0] ** 4 + x[1] ** 2 - x[0] ** 2 * x[1]])

    def eq_con(x, kkt):
        return np.array([1 - 2 * x[0] * x[1] / 3, (3 * x[0] ** 2 - 4 * x[1]) / 3 + 1])

    x0 = np.array([0.5, 3.0])
    nil = np.array([])
    out = minimize(J, x0, nonlconeq=eq_con, method='SQP')
    assert abs(out['x'][0] - 1) < tol
    assert abs(out['x'][1] - 1.5) < tol
    assert abs(out['grad'][0] - 1) < tol
    assert abs(out['grad'][1] - 2) < tol
    assert abs(out['kkt'].equality_nonlinear[0] - 2) < tol
    assert abs(out['kkt'].equality_nonlinear[1] - 0.5) < tol

def test_sqp2():
    def J(x):
        return np.array([x[0] ** 4 + x[1] ** 2 - x[0] ** 2 * x[1]])

    def eq_con(x, l):
        return np.array([1 - 2 * x[0] * x[1] / 3, (3 * x[0] ** 2 - 4 * x[1]) / 3 + 1])

    x0 = np.array([0.5, 3.0])
    nil = np.array([])
    out = minimize(J, x0, nonlconineq=eq_con, method='SQP')
    assert abs(out['x'][0] - 1) < tol
    assert abs(out['x'][1] - 1.5) < tol
    assert abs(out['grad'][0] - 1) < tol
    assert abs(out['grad'][1] - 2) < tol
    assert abs(out['kkt'].inequality_nonlinear[0] - 2) < tol
    assert abs(out['kkt'].inequality_nonlinear[1] - 0.5) < tol

from npnlp import minimize
import numpy as np

def J(x):
    return np.array([x[0]**4 + x[1]**2 - x[0]**2*x[1]])

def eq_con(x):
    return np.array([1 - 2*x[0]*x[1]/3, (3*x[0]**2 - 4*x[1])/3 + 1])

x0 = np.array([0.5, 3.0])

nil = np.array([])

out = minimize(J, x0, A=nil, b=nil, Aeq=nil, beq=nil, lb=nil, ub=nil, nonlconineq=eq_con, method='SQP')

print(out)

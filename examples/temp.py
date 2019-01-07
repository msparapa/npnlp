from npnlp import minimize
import numpy as np

n = 5


def J(x):
    return np.sum(x)

x0 = np.zeros(5)

out = minimize(J, x0, method='SQP')

print(out)

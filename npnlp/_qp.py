from .npnlp import npnlp_result
import numpy as np
from scipy.linalg import null_space, cholesky, inv

def qp(G, d, Aeq, beq):
    """
    Solves the quadratic subproblem. Note: This really just calls numpy's linalg solve method right now.

    min_x J = 1/2 x' G x + x'd
    subject to: Aeq x = beq

    :param G:
    :param d:
    :param Aeq:
    :param beq:
    :return:
    """
    opt = npnlp_result()
    x_dim = G.shape[0]
    zeros_shape = (Aeq.shape[0],) * 2
    K = np.block([[G, Aeq.T], [Aeq, np.zeros(zeros_shape)]])
    ans = np.linalg.solve(K, np.vstack((-d, beq)))
    opt['x'] = ans[:x_dim]
    opt['lagrange'] = ans[x_dim:]
    opt['fval'] = 1
    opt['exitflag'] = 0
    opt['message'] = 'Optimization terminated successfully.'
    opt['grad'] = G.dot(opt['x']) + d
    opt['hessian'] = G
    opt['success'] = True
    opt['nfev'] = 1
    opt['ngev'] = 1
    opt['nhev'] = 1
    opt['nit'] = 1
    opt['maxconstrv'] = max(Aeq.dot(opt['x']) - beq)

    return opt
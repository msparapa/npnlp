from .npnlp import npnlp_result
import numpy as np
from scipy.optimize.slsqp import approx_jacobian

def sqp(costfun, x0, lagrange0, A, b, Aeq, beq, lb, ub, nonlconeq, nonlconineq, **kwargs):
    """

    :param costfun:
    :param x0:
    :param lagrange0:
    :param A:
    :param b:
    :param Aeq:
    :param beq:
    :param lb:
    :param ub:
    :param nonlconeq:
    :param nonlconineq:
    :return:
    """
    max_super_iterations = kwargs.get('max_super_iterations', 100)

    opt = npnlp_result()

    if A is None:
        A = np.array([])

    if b is None:
        b = np.array([])

    if Aeq is None:
        Aeq = np.array([])

    if beq is None:
        beq = np.array([])

    n_states = x0.shape[0]

    if nonlconeq is None:
        n_lconeq = 0
    else:
        n_lconeq = nonlconeq(x0).shape[0]

    if nonlconineq is None:
        n_lconineq = 0
    else:
        n_lconineq = nonlconineq(x0).shape[0]

    if lagrange0 is None:
        lagrange0 = np.zeros(n_lconeq + n_lconineq)

    if n_lconeq == 0 and n_lconineq == 0:
        def lagrangian(x,l):
            return costfun(x)
    elif n_lconineq == 0:
        def lagrangian(x,l):
            return costfun(x) + (l.T).dot(nonlconeq(x))
    elif n_lconeq == 0:
        raise NotImplementedError
    else:
        raise NotImplementedError

    nsuperit = 0
    running = True
    converged = False

    xk = np.copy(x0)
    lagrangek = np.copy(lagrange0)
    B = np.eye(n_states) # Start with B = I to ensure positive definite
    from ._qp import qp
    while running:
        # Min Q(s) = f(x) + df(x)' s + 1/2 s' B s
        # subject to dg_j(x)' s + delta_j g_j(x) <= 0
        #            dh_k(x)' s + deltabar h_k(x) = 0
        # B is second derivative of L
        # L(x) = f(x) + sum_j lambda_j * g_j(x) + sum_k lambda_(k+m) h_k(x) ; for m equality constraints
        # delta_j = 1 if g_j(x) < 0 (satisfied constraint)
        # delta_j = deltabar if g_j(x) >= 0 (active/violated constraint)
        # 0 < deltabar <= 1.0 very common to use deltabar = 0.9 to 0.95
        # Handle linear constraints seperately
        # d g_j(x)' s + 1 g_j(x) <= 0
        # d h_k(x)' s + 1 h_k(x) = 0
        # Basically delta_j = 1 for all
        # d Q(s) = d f(x) + B s
        # Find s s.t. d Q(s) = 0
        # QP solution should give s and lambda (use lambda from QP to update lambda in SQP)
        # Step size problem, use modified lagrangian function
        # u_j are lagrange multipliers from the f(x) problem (lambdas from the QP problem)
        # phi(alpha) = f(xk + alpha s) + sum_j u_j max(0, g_j(xk, + alpha s)) + sum_k u_(k+m) |h_k(xk + alpha s)|
        # xk known, s known
        # First iteration:
        # u_j = |lambda_j| at QP solution for s_1
        # solve phi, inexact line search
        # xk1 = xk + alpha s_1
        # Iterations 2 and beyond
        # u_j = max(|lambda_j|, 1/2(u'_j + |lambda_j|)
        # lambda_j's just came from the QP problem, u'_j is previous value of u_j
        # Update B
        # B_new = B - (B p p' B)/(p' B p) + (eta eta') / (p' eta)
        # p = xk1 - xk
        # p = alpha s
        # eta = theta y + (1- theta) B p
        # theta is variable metric (0 <= theta <= 1)
        # Theta can change each B update
        # Theta = 1 if p' y >= (0.2) p' B p
        # Theta = ((0.8) p' B p) / (p' B p - p' y) otherwise
        # (0.2) and (0.8) are experience based values
        # y = d_x L(xk1, lk) - d_x L(xk, lk)
        # At constrained minimum, 3rd KT condition
        # d_x L = 0
        A = approx_jacobian(xk, nonlconeq, 1e-6)
        d = approx_jacobian(xk, costfun, 1e-6).T
        # W = approx_jacobian(xk, lambda _: approx_jacobian(_, costfun, 1e-6)[0], 1e-6)
        breakpoint()
        # opt = qp(W,d,A,np.zeros((A.shape[0],1))) # Solve QP
        # Update B, B eventually becomes d^2 L
        xk = np.copy(opt['x'].T[0])
        lagrangek = np.copy(opt['lagrange'].T[0])
        nsuperit += 1
        if nsuperit > max_super_iterations:
            running = False

    opt['nsuperit'] = nsuperit

    return opt

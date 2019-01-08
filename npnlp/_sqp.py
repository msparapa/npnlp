from .npnlp import npnlp_result
import numpy
import cvxopt
from scipy.optimize import fminbound
from scipy.optimize.slsqp import approx_jacobian

def sqp(costfun, x0, lagrange0, A, b, Aeq, beq, lb, ub, nonlconeq, nonlconineq, **kwargs):
    """
    A sequential quadratic programming solver. Higher level solver manipulates the problem using
    NumPy and SciPy while the QP subproblems are solved using cvxopt.

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
    delta_bar = kwargs.get('delta_bar', 0.90)
    max_super_iterations = kwargs.get('max_super_iterations', 100)
    tolerance = kwargs.get('tolerance', 1e-6)

    epsilon = tolerance

    opt = npnlp_result()

    if A is None:
        A = numpy.array([])

    if b is None:
        b = numpy.array([])

    if Aeq is None:
        Aeq = numpy.array([])

    if beq is None:
        beq = numpy.array([])

    n_states = x0.shape[0]

    if nonlconeq is None:
        n_lconeq = 0
        def nonlconeq(*args):
            return numpy.array([])
    else:
        n_lconeq = nonlconeq(x0).shape[0]

    if nonlconineq is None:
        n_lconineq = 0
        def nonlconineq(*args):
            return numpy.array([])
    else:
        n_lconineq = nonlconineq(x0).shape[0]

    if lagrange0 is None:
        lagrange0 = numpy.zeros(n_lconeq + n_lconineq)

    if n_lconeq == 0 and n_lconineq == 0:
        def lagrangian(x,l):
            return costfun(x)
    elif n_lconineq == 0:
        def lagrangian(x,l):
            return costfun(x) + (l.T).dot(nonlconeq(x))
    elif n_lconeq == 0:
        def lagrangian(x,l):
            return costfun(x) + (l.T).dot(nonlconineq(x))
    else:
        raise NotImplementedError

    nsuperit = 0
    running = True
    converged = False

    xk = numpy.copy(x0)
    lagrangek = numpy.copy(lagrange0)
    B = numpy.eye(n_states) # Start with B = I to ensure positive definite
    from ._qp import qp
    while running:
        f = costfun(xk)
        df = approx_jacobian(xk, costfun, epsilon)
        # Min Q(s) = f(x) + df(x)' s + 1/2 s' B s
        # subject to dg_j(x)' s + delta_j g_j(x) <= 0
        #            dh_k(x)' s + deltabar h_k(x) = 0
        # B is second derivative of L
        # L(x) = f(x) + sum_j lambda_j * g_j(x) + sum_k lambda_(k+m) h_k(x) ; for m equality constraints

        # delta_j = 1 if g_j(x) < 0 (satisfied constraint)
        # delta_j = deltabar if g_j(x) >= 0 (active/violated constraint)
        if n_lconineq > 0:
            dg = approx_jacobian(xk, nonlconineq, epsilon)
            nonlconineq_eval = nonlconineq(xk)
            nonlconineq_satisfied = nonlconineq_eval < 0
            delta = numpy.array([1 if satisfied else delta_bar for satisfied in nonlconineq_satisfied])
        else:
            nonlconineq_eval = numpy.array([])

        if n_lconeq > 0:
            dh = approx_jacobian(xk, nonlconeq, epsilon)
            nonlconeq_eval = nonlconeq(xk)
            nonlconeq_satisfied = numpy.abs(nonlconeq_eval) < tolerance
        else:
            nonlconeq_eval = numpy.array([])

        P = cvxopt.matrix(B)
        q = cvxopt.matrix(df[0])
        if n_lconineq > 0:
            G = cvxopt.matrix(dg)
            h = cvxopt.matrix(-delta*nonlconineq_eval)
        else:
            G = None
            h = None

        if n_lconeq > 0:
            A = cvxopt.matrix(dh)
            b = cvxopt.matrix(-delta_bar*nonlconeq_eval)
        else:
            A = None
            b = None

        # TODO: Handle linear constraints seperately
        # d g_j(x)' s + 1 g_j(x) <= 0
        # d h_k(x)' s + 1 h_k(x) = 0
        # Basically delta_j = 1 for all
        # QP solution should give s and lambda (use lambda from QP to update lambda in SQP)

        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        s = numpy.array(sol['x']).T[0]
        lagrange_eq = numpy.array(sol['y']).T[0] # lambda eq
        lagrange_ineq = numpy.array(sol['z']).T[0] # Lambda ineq
        lagrangek1 = numpy.hstack((lagrange_eq, lagrange_ineq))

        # First iteration:
        # u_j = |lambda_j| at QP solution for s_1
        # solve phi, inexact line search
        # xk1 = xk + alpha s_1
        # Iterations 2 and beyond
        # u_j = max(|lambda_j|, 1/2(u'_j + |lambda_j|)
        # lambda_j's just came from the QP problem, u'_j is previous value of u_j

        if nsuperit == 0:
            uineq = abs(lagrange_ineq)
        else:
            uineq = numpy.maximum(abs(lagrange_ineq), 1/2*(uineq + abs(lagrange_ineq)))

        # Get step size
        def phi(alpha):
            return costfun(xk + alpha*s) + numpy.inner(uineq, numpy.maximum(0, nonlconineq(xk + alpha*s))) + numpy.inner(lagrange_eq, numpy.maximum(0, abs(nonlconeq(xk + alpha*s))))

        alpha = fminbound(phi, 0, 2)
        p = alpha * s
        xk1 = xk + p
        y = approx_jacobian(xk1, lambda _: lagrangian(_, lagrangek) - lagrangian(xk, lagrangek), epsilon)[0]

        if numpy.inner(p, y) >= 0.2*B.dot(p).dot(p):
            theta = 1
        else:
            theta = 0.8 * B.dot(p).dot(p)/(B.dot(p).dot(p) - numpy.inner(p, y))

        eta = theta * y + (1 - theta)*B.dot(p)

        # BFGS update of the hessian
        Bk1 = B - B.dot(numpy.outer(p,p)).dot(B) / (B.dot(p).dot(p)) + numpy.dot(eta, eta.T) / eta.dot(p)
        print('etas not correct :(')
        breakpoint()
        change = xk1 - xk
        xk = numpy.copy(xk1)
        lagrangek = numpy.copy(lagrangek1)
        nsuperit += 1

        if all(abs(change) < tolerance):
            running = False

        if nsuperit > max_super_iterations:
            running = False

    if all(abs(nonlconeq_eval) < tolerance) and all(nonlconineq_eval < tolerance):
        converged = True

    opt['x'] = xk
    opt['fval'] = 9999999999
    opt['success'] = converged
    opt['lagrange'] = lagrangek
    opt['grad'] = approx_jacobian(xk, costfun, epsilon)
    opt['hessian'] = B
    opt['nsuperit'] = nsuperit

    return opt

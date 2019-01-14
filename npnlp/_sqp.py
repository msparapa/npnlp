from .npnlp import npnlp_result, kkt_multipliers
import numpy
import cvxopt
from scipy.optimize import fminbound
from scipy.optimize.slsqp import approx_jacobian
from scipy.linalg import lu


def sqp(costfun, x0, kkt0, A, b, Aeq, beq, lb, ub, nonlconeq, nonlconineq, **kwargs):
    """
    A sequential quadratic programming solver. Higher level solver manipulates the problem using
    NumPy and SciPy while the QP subproblems are solved using cvxopt.

    :param costfun:
    :param x0:
    :param kkt0:
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
    tolerance = kwargs.get('tolerance', 1e-8)

    epsilon = tolerance

    opt = npnlp_result()

    if A is None:
        A = numpy.array([])
    else:
        A = A.astype(numpy.float64)

    if b is None:
        b = numpy.array([])
    else:
        b = b.astype(numpy.float64)

    if Aeq is None:
        Aeq = numpy.array([])
    else:
        Aeq = Aeq.astype(numpy.float64)

    if beq is None:
        beq = numpy.array([])
    else:
        beq = beq.astype(numpy.float64)

    n_states = x0.shape[0]

    if kkt0 is None:
        kkt0 = kkt_multipliers()

    if nonlconeq is None:
        n_lconeq = 0
        def nonlconeq(*args):
            return numpy.array([])
    else:
        n_lconeq = nonlconeq(x0, kkt0).shape[0]

    if nonlconineq is None:
        n_lconineq = 0
        def nonlconineq(*args):
            return numpy.array([])
    else:
        n_lconineq = nonlconineq(x0, kkt0).shape[0]

    if n_lconeq == 0 and n_lconineq == 0:
        def lagrangian(x, kkt):
            return costfun(x)
    elif n_lconineq == 0:
        def lagrangian(x, kkt):
            return costfun(x) + numpy.inner(kkt.equality_nonlinear, nonlconeq(x, kkt))
    elif n_lconeq == 0:
        def lagrangian(x, kkt):
            return costfun(x) + numpy.inner(kkt.inequality_nonlinear, nonlconineq(x, kkt))
    else:
        def lagrangian(x, kkt):
            return costfun(x) + numpy.inner(kkt.equality_nonlinear, nonlconeq(x, kkt)) + numpy.inner(kkt.inequality_nonlinear, nonlconineq(x, kkt))

    nsuperit = 0
    nit = 0
    running = True
    converged = False

    xk = numpy.copy(x0)
    kktk = kkt_multipliers()
    kktk.equality_linear = numpy.copy(kkt0.equality_linear)
    kktk.equality_nonlinear = numpy.copy(kkt0.equality_nonlinear)
    kktk.inequality_linear = numpy.copy(kkt0.inequality_linear)
    kktk.inequality_nonlinear = numpy.copy(kkt0.inequality_nonlinear)
    kktk.lower = numpy.copy(kkt0.lower)
    kktk.upper = numpy.copy(kkt0.upper)
    B = numpy.eye(n_states) # Start with B = I to ensure positive definite
    cvxopt.solvers.options['show_progress'] = False
    while running:
        f = costfun(xk)
        df = approx_jacobian(xk, costfun, epsilon)

        if n_lconineq > 0:
            dg = approx_jacobian(xk, lambda _: nonlconineq(_, kktk), epsilon)
            nonlconineq_eval = nonlconineq(xk, kktk)
            nonlconineq_satisfied = nonlconineq_eval < 0
            delta = numpy.array([1 if satisfied else delta_bar for satisfied in nonlconineq_satisfied])
        else:
            nonlconineq_eval = numpy.array([])

        if n_lconeq > 0:
            dh = approx_jacobian(xk, lambda _: nonlconeq(_, kktk), epsilon)
            nonlconeq_eval = nonlconeq(xk, kktk)
            nonlconeq_satisfied = numpy.abs(nonlconeq_eval) < tolerance
        else:
            nonlconeq_eval = numpy.array([])

        q = df[0]
        if n_lconineq > 0 and len(A) > 0:
            G = cvxopt.matrix(numpy.hstack((dg, A)))
            h = cvxopt.matrix(numpy.hstack((-delta*nonlconineq_eval, b)))
        elif n_lconineq > 0:
            G = cvxopt.matrix(dg)
            h = cvxopt.matrix(-delta*nonlconineq_eval)
        elif len(A) > 0:
            G = cvxopt.matrix(A)
            h = cvxopt.matrix(b)
        else:
            G = None
            h = None

        if n_lconeq > 0 and len(Aeq) > 0:
            R = numpy.hstack((dh, Aeq))
            t = numpy.hstack((-delta_bar*nonlconeq_eval, beq))
        elif n_lconeq > 0:
            R = dh
            t = -delta_bar*nonlconeq_eval
        elif len(Aeq) > 0:
            R = Aeq
            t = beq
        else:
            R = None
            t = None

        rshape = numpy.linalg.matrix_rank(R)
        reshaped = False
        if numpy.linalg.matrix_rank(R) < R.shape[0]:
            reshaped = True
            pshape = B.shape[0]
            Q, L, U = lu(numpy.column_stack((R, t)))
            M = L.dot(U)
            R = M[:rshape, :-1]
            t = M[:rshape, -1]
            Rfull = M[:,:-1]
            Momit = M[rshape:,:]
            tfull = M[:, -1]

        try:
            qpsol = cvxopt.solvers.qp(cvxopt.matrix(B), cvxopt.matrix(q), G, h, cvxopt.matrix(R), cvxopt.matrix(t))
        except:
            qpsol['status'] = 'Nope!'


        s = numpy.array(qpsol['x']).T[0]
        kktk.equality_linear = numpy.array(qpsol['y']).T[0][:len(beq)]
        kktk.equality_nonlinear = numpy.array(qpsol['y']).T[0][len(beq):]
        kktk.inequality_linear = numpy.array(qpsol['z']).T[0][:len(b)]
        kktk.inequality_nonlinear = numpy.array(qpsol['z']).T[0][len(b):]
        la = kktk.equality_nonlinear

        if reshaped:
            # TODO: This entire "reshaping" can be done using pure LA and no goofy if-then statements. IDK how though ATM
            la_append = None
            for rrow in range(Momit.shape[0]):
                ind = numpy.argmax([numpy.inner(M[row,:], Momit[rrow,:])/(numpy.linalg.norm(M[row,:])*numpy.linalg.norm(Momit[rrow,:])) for row in range(rshape)])
                if la_append is None:
                    la_append = la[ind]
                else:
                    la_append = numpy.hstack((la_append, la[ind]))

            la = numpy.hstack((la, la_append))
            kktk.equality_nonlinear = Q.dot(la)

        nit += qpsol['iterations']

        if nsuperit == 0:
            uineq = abs(kktk.inequality_nonlinear)
            ueq = abs(kktk.equality_nonlinear)
        else:
            uineq = numpy.maximum(abs(kktk.inequality_nonlinear), 1 / 2*(uineq + abs(kktk.inequality_nonlinear)))
            ueq = numpy.maximum(abs(kktk.equality_nonlinear), 1 / 2 * (ueq + abs(kktk.equality_nonlinear)))

        # Get step size
        def phi(alpha):
            return costfun(xk + alpha*s) + numpy.inner(uineq, numpy.maximum(0, nonlconineq(xk + alpha*s, kktk))) +\
                   numpy.inner(ueq, abs(nonlconeq(xk + alpha*s, kktk)))

        alpha = fminbound(phi, 0, 2) # TODO: Replace this with a more efficient inexact line search
        p = alpha * s
        xk1 = xk + p

        y = (approx_jacobian(xk1, lambda _: lagrangian(_, kktk), epsilon) - approx_jacobian(xk, lambda _: lagrangian(_, kktk), epsilon))[0]

        # Damped BFGS update of the hessian
        if numpy.inner(p, y) >= 0.2*B.dot(p).dot(p):
            theta = 1
        else:
            theta = 0.8 * B.dot(p).dot(p)/(B.dot(p).dot(p) - numpy.inner(p, y))

        eta = theta * y + (1 - theta)*B.dot(p)

        B = B - B.dot(numpy.outer(p,p)).dot(B) / (B.dot(p).dot(p)) + numpy.outer(eta, eta) / eta.dot(p)

        change = xk1 - xk
        xk = numpy.copy(xk1)
        nsuperit += 1

        if all(abs(change) < tolerance):
            running = False

        if nsuperit > max_super_iterations:
            running = False

    if all(abs(nonlconeq_eval) < tolerance) and all(nonlconineq_eval < tolerance) and qpsol['status'] == 'optimal':
        converged = True

    opt['x'] = xk
    opt['fval'] = costfun(xk)
    opt['success'] = converged
    opt['kkt'] = kktk
    opt['grad'] = approx_jacobian(xk, costfun, epsilon)[0]
    opt['hessian'] = B
    opt['nsuperit'] = nsuperit
    opt['nit'] = nit

    return opt

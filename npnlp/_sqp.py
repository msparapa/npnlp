from .npnlp import npnlp_result

def sqp(costfun, x0, A, b, Aeq, beq, lb, ub, nonlconeq, nonlconineq, options):
    opt = npnlp_result()
    return opt
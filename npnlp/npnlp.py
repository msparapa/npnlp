methods = ['sqp']

class npnlp_result(dict):
    def __new__(cls, *args, **kwargs):
        obj = super(npnlp_result, cls).__new__(cls, *args, **kwargs)
        obj2 = {'x':None,
                'fval':None,
                'exitflag':None,
                'message':None,
                'lagrange':None,
                'grad':None,
                'hessian':None,
                'success':None,
                'nfev':None,
                'ngev':None,
                'nhev':None,
                'nit':None,
                'nsuperit':None,
                'maxconstv':None}
        obj.update(obj2)
        return obj

    def __str__(self):
        out = ''
        for key in self.keys():
            out += str(key) + ': ' + str(self[key]) + '\n'
        return out

def minimize(costfun, x0=None, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None, nonlconeq=None, nonlconineq=None, options=None, method=None):
    if method is None:
        raise Exception('\'method\' must be defined.')

    if not isinstance(method, str):
        raise Exception('\'method\' must be a string.')

    if method.lower() not in methods:
        raise Exception('Unknown method \'' + method.lower() + '\'')

    if method.lower() == 'sqp':
        from ._sqp import sqp
        opt = sqp(costfun, x0, A, b, Aeq, beq, lb, ub, nonlconeq, nonlconineq, options)

    return opt

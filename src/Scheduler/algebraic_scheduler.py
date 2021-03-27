from math import exp
from .func_scheduler import FunctionScheduler


class ExpScheduler(FunctionScheduler):
    def __init__(self, e_offset=0.0, e_scale=1.0, decay_rate=0.1, auto_scale=True, **kwargs):
        def exp_func(x):
            """
            Exponential function of the form N =  No*e^(-lambda*x) + C
            """
            return e_scale*exp(-decay_rate*x) + e_offset
        super(ExpScheduler, self).__init__(func=exp_func, auto_scale=auto_scale, **kwargs)


class LinearScheduler(FunctionScheduler):
    def __init__(self, l_offset=0.0, l_slope=1.0, auto_scale=False, **kwargs):
        def linear_func(x):
            """
            Linear function of the form y = m*x + c
            """
            return l_slope * x + l_offset
        super(LinearScheduler, self).__init__(func=linear_func, auto_scale=auto_scale, **kwargs)

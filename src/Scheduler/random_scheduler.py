from numpy import random
from .func_scheduler import FunctionScheduler


class RandomNormalScheduler(FunctionScheduler):
    def __init__(self, mu=0.0, std=1.0, clip_range=None, sample_size=None, auto_scale=False, **kwargs):
        if clip_range is None:
            clip_range = (-1.0, 1.0)
        def normal_func(x):
            return random.normal(mu, std, size=sample_size)
        super(RandomNormalScheduler, self).__init__(func=normal_func, auto_scale=auto_scale, clip_range=clip_range, **kwargs)


class RandomUniformScheduler(FunctionScheduler):
    def __init__(self, clip_range=None, sample_size=None, auto_scale=False, **kwargs):
        if clip_range is None:
            clip_range = (-1.0, 1.0)

        def uniform_func(x):
            return random.uniform(*clip_range, size=sample_size)
        
        super(RandomUniformScheduler, self).__init__(func=uniform_func, auto_scale=auto_scale, clip_range=clip_range, **kwargs)
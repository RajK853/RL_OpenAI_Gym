import numpy as np


class BaseScheduler:
    def __init__(self, *, clip_range, update_step=1):
        self.clip_range = clip_range
        self.update_step = update_step
        self.value = 0.0
        self.steps = 0

    def calculate_value(self):
        raise NotImplementedError

    def update_value(self):
        val = self.calculate_value()
        self.value = np.clip(val, *self.clip_range)

    def increment(self, step=1):
        self.steps += step
        if (self.steps % self.update_step) == 0:
            self.update_value()

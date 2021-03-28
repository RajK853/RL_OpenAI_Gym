import numpy as np
from .base_algorithm import BaseAlgorithm


class OnPolicyAlgorithm(BaseAlgorithm):

    def __init__(self, batch_size=None, **kwargs):
        super(OnPolicyAlgorithm, self).__init__(batch_size=batch_size, **kwargs)
        self.field_names = ()
        self.trajectory = {}

    def add_transition(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.trajectory.keys():
                self.trajectory[key] = []
            self.trajectory[key].append(value)

    def sample_trajectory(self):
        assert self.field_names, "Attribute field_names not defined!"
        return (np.array(self.trajectory[key]) for key in self.field_names)

    def clear_trajectory(self):
        for array in self.trajectory.values():
            array.clear()

    def train(self):
        raise NotImplementedError

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        if self.training:
            self.add_summaries(self.epoch)
            self.trajectory.clear()

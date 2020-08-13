import numpy as np
from src.Buffer import ReplayBuffer
from .base_algorithm import BaseAlgorithm


class OffPolicyAlgorithm(BaseAlgorithm):

    def __init__(self, *, buffer_size=100000, **kwargs):
        super(OffPolicyAlgorithm, self).__init__(**kwargs)
        self.replay_buffer = ReplayBuffer(size=buffer_size)
        self.scalar_summaries += ("buffer_size", )

    @property
    def buffer_size(self):
        return len(self.replay_buffer)

    def train(self):
        raise NotImplementedError

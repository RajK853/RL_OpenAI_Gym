import numpy as np
from collections import deque
from .replay_buffer import ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size):
        super().__init__(size)
        self._priorities = deque(maxlen=size)

    def add(self, state, action, reward, obs_t1, done, **kwargs):
        super().add(state, action, reward, obs_t1, done)
        self._priorities.append(max(self._priorities, default=1))

    def sample(self, batch_size, priority_scale=1.0):
        batch_size = min(len(self), batch_size)
        batch_probs = self.get_probabilities(priority_scale)
        batch_indices = np.random.choice(range(len(self)), size=batch_size, p=batch_probs)
        batch_importance = self.get_importance(batch_probs[batch_indices])
        batch = np.array(self._storage)[batch_indices].T
        return batch, batch_importance, batch_indices

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self._priorities) ** priority_scale
        batch_probabilities = scaled_priorities / sum(scaled_priorities)
        return batch_probabilities

    def get_importance(self, probabilities):
        importance = 1 / (len(self) * probabilities)  # TODO: The change here might create problem
        importance_normalized = importance / max(importance)
        return importance_normalized

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self._priorities[i] = abs(e) + offset

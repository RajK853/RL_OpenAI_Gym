import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Simple replay buffer to store and sample transition experiences
    """

    def __init__(self, size):
        """
        Constructor function
        args:
            size (int) : Maximum size of replay buffer
        """
        self._maxsize = size
        self._pointer = 0
        self._size = 0
        self._buffer = None

    def __len__(self):
        return self._size

    def _init_buffer(self, sample):
        buffer = []
        for obj in sample:
            shape = (self._maxsize, )
            if isinstance(obj, np.ndarray):
                shape += obj.shape
                dtype = obj.dtype
            else:
                dtype = type(obj)
            buffer.append(np.zeros(shape, dtype=dtype))
        self._buffer = buffer
        self._size = 0
        self._pointer = 0

    def add(self, transition):
        """
        Add transition data to the replay buffer
        """
        if self._buffer is None:
            self._init_buffer(transition)
        i = self._pointer
        for obj, buffer in zip(transition, self._buffer):
            buffer[i] = obj
        self._pointer = (self._pointer + 1) % self._maxsize
        self._size = min(self._size + 1, self._maxsize)

    def sample(self, batch_size):
        """
        Sample data from the replay buffer
        args:
            batch_size (int) : Maximum batch size to sample
        returns:
            tuple of 5 lists : Sampled batch of transitions 
        """
        # batch_size = min(batch_size, len(self))
        idxes = np.random.randint(0, len(self), size=batch_size)
        return tuple(buffer[idxes] for buffer in self._buffer)

    def sample_all(self):
        idxes = np.arange(self._size)
        return tuple(buffer[idxes] for buffer in self._buffer)        

    def clear(self):
        """
        Clear the contents of replay buffer
        """
        self._buffer.clear()
        self._size = 0
        self._pointer = 0


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size):
        super().__init__(size)
        self._priorities = deque(maxlen=size)

    def add(self, transition):
        super().add(transition)
        self._priorities.append(max(self._priorities, default=1))

    def sample(self, batch_size, priority_scale=1.0):
        batch_size = min(len(self), batch_size)
        batch_probs = self.get_probabilities(priority_scale)
        batch_indices = np.random.choice(range(len(self)), size=batch_size, p=batch_probs)
        batch_importance = self.get_importance(batch_probs[batch_indices])
        batch = np.array(self._buffer)[batch_indices].T
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

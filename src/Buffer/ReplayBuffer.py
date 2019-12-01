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
        self._storage = deque(maxlen=size)

    def __len__(self):
        return len(self._storage)

    def add(self, transition):
        """
        Add transition data to the replay buffer
        args:
            state : Current state
            action : Action taken
            reward (float) : Received reward
            next_state : Next state
            done (bool) : Episode done
        """
        assert len(transition) == 5, "Invalid data received by the replay buffer; {}".format(transition)
        self._storage.append(transition)

    def _encode_sample(self, idxes):
        """
        Sample data from given indexes
        args:
            idxes (list/np.array) : List with indexes of data to sample
        returns:
            np.array, np.array, np.array, np.array, np.array : Sampled states, actions, rewards, next_states and dones
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idxes:
            obs_t, action, reward, obs_tp1, done = self._storage[i]
            states.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def sample(self, batch_size):
        """
        Sample data from the replay buffer
        args:
            batch_size (int) : Maximum batch size to sample
        returns:
            tuple of 5 lists : Sampled batch of transitions 
        """
        batch_size = min(len(self), batch_size)
        idxes = np.random.randint(0, len(self), size=batch_size)
        return self._encode_sample(idxes)

    def clear(self):
        """
        Clear the contents of replay buffer
        """
        self._storage.clear()


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

import numpy as np
from . import OnPolicyAlgorithm
from src.utils import get_scheduler, standardize_array


class Reinforce(OnPolicyAlgorithm):
    VALID_POLICIES = {"DiscretePolicy", "GaussianPolicy"}

    def __init__(self, *, gamma_kwargs, num_train=1, **kwargs):
        super(Reinforce, self).__init__(**kwargs)
        self.gamma_scheduler = get_scheduler(gamma_kwargs)
        self.num_train = num_train
        self.field_names = ("state", "action", "reward")    # Used to sample out the trajectory
        self.scalar_summaries += ("gamma", )

    @property
    def gamma(self):
        return self.gamma_scheduler.value

    def compute_discounted_return(self, rewards):
        discounted_return = 0
        returns = np.zeros(self.epoch_length, np.float32)
        for i in reversed(range(self.epoch_length)):
            discounted_return = rewards[i] + self.gamma * discounted_return
            returns[i] = discounted_return
        return returns

    def train(self):
        states, actions, rewards = self.sample_trajectory()
        discounted_return = self.compute_discounted_return(rewards)
        discounted_return = standardize_array(discounted_return)
        discounted_return = np.expand_dims(discounted_return, axis=-1)
        feed_dict = {self.policy.model.input: states, self.policy.actions_ph: actions,
                     self.policy.targets_ph: discounted_return}
        for _ in range(self.num_train):
            self.policy.update(self.sess, feed_dict=feed_dict)

    def hook_before_train(self, **kwargs):
        self.policy.init_default_loss()
        super().hook_before_train(**kwargs)

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        if self.training:
            self.gamma_scheduler.increment()

    def hook_after_step(self, **kwargs):
        super().hook_after_step(**kwargs)
        if self.training:
            state, action, reward, *_ = self.transition
            self.add_transition(state=state, action=action, reward=reward)

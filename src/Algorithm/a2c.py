import numpy as np
from .reinforce import Reinforce
from src.Layer import QNetwork


class A2C(Reinforce):

    def __init__(self, **kwargs):
        super(A2C, self).__init__(**kwargs)
        self.critic = QNetwork(self.obs_shape, output_size=1, layer_units=(50, 50), scope="critic")
        self.mean_critic_loss = None
        self.summary_init_objects += (self.critic, )
        self.scalar_summaries += ("critic_loss", )

    def hook_after_epoch(self, **kwargs):
        super(Reinforce, self).hook_after_epoch(**kwargs)
        if self.training:
            states, actions, rewards = self.sample_trajectory()
            discounted_return = self.compute_discounted_return(rewards)
            values = self.critic.predict(sess=self.sess, inputs=states)
            discounted_return = discounted_return - values.squeeze()
            self.mean_policy_loss = np.mean([self.policy.update(self.sess, states, actions, targets=discounted_return)
                                             for _ in range(self.num_train)])
            # TODO: Refactor this
            discounted_return = discounted_return.reshape((*discounted_return.shape, 1))
            self.mean_critic_loss = np.mean([self.critic.update(self.sess, states, targets=discounted_return)
                                             for _ in range(self.num_train)])
            self.add_summaries()
            self.trajectory.clear()

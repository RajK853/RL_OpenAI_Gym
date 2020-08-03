import numpy as np
from .reinforce import Reinforce
from src.Layer import QNetwork


class ActorCritic(Reinforce):

    def __init__(self, **kwargs):
        super(ActorCritic, self).__init__(**kwargs)
        self.critic = QNetwork(self.obs_shape, output_size=1, layer_units=(50, 50), scope="critic")
        self.mean_critic_loss = None
        self.summary_init_objects += (self.critic, )
        self.scalar_summaries += ("critic_loss", )

    def calculate_advantage(self, states, rewards):
        discounted_return = self.compute_discounted_return(rewards)
        values = self.critic.predict(sess=self.sess, inputs=states)
        advantage = discounted_return - values.squeeze()
        return advantage

    def train_actor(self, states, actions, advantage):
        self.mean_policy_loss = np.mean([self.policy.update(self.sess, states, actions, targets=advantage)
                                         for _ in range(self.num_train)])

    def train_critic(self, states, advantage):
        # TODO: Refactor this to not use reshape
        advantage = advantage.reshape((*advantage.shape, 1))
        self.mean_critic_loss = np.mean([self.critic.update(self.sess, states, targets=advantage)
                                         for _ in range(self.num_train)])

    def train(self):
        states, actions, rewards = self.sample_trajectory()
        advantage = self.calculate_advantage(states, rewards)
        self.train_actor(states, actions, advantage)
        self.train_critic(states, advantage)

    """def hook_after_epoch(self, **kwargs):
        super(Reinforce, self).hook_after_epoch(**kwargs)
        if self.training:
            self.train()
            self.add_summaries()
            self.trajectory.clear()"""

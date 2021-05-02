import numpy as np
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf_v1

from . import OnPolicyAlgorithm
from src.utils import get_scheduler, standardize_array
from src.Network.utils import get_clipped_train_op

tfd = tfp.distributions


class Reinforce(OnPolicyAlgorithm):
    VALID_POLICIES = {"DiscretePolicy", "GaussianPolicy"}

    def __init__(self, *, alpha=0.001, gamma_kwargs, **kwargs):
        super(Reinforce, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma_scheduler = get_scheduler(gamma_kwargs)
        self.schedulers += (self.gamma_scheduler, )
        self.targets_ph = tf_v1.placeholder("float32", shape=(None, 1), name="target_ph")
        self.states_ph = tf_v1.placeholder("float32", shape=[None, *self.obs_shape], name="states_ph")
        if self.policy_type == "DiscretePolicy":
            self.actions_ph = tf_v1.placeholder("int32", shape=(None, ), name=f"actions_ph")
        else:
            self.actions_ph = tf_v1.placeholder("float32", shape=(None, *self.action_shape), name=f"actions_ph")
        self.field_names = ("state", "action", "reward")    # Used to sample out the trajectory
        self.scalar_summaries += ("gamma", )

    @property
    def gamma(self):
        return self.gamma_scheduler.value

    def init_actor_loss(self):
        if self.policy_type == "GaussianPolicy":
            mu, std = self.policy.mu_and_std(self.states_ph)
            norm_dist = tfd.Normal(loc=mu, scale=std)
            entropy = norm_dist.entropy()
            log_actions = norm_dist.log_prob(tf_v1.atanh(self.actions_ph))
            log_actions -= tf_v1.log(1.0 - self.actions_ph**2 + 1e-8)
            log_actions = tf_v1.reduce_sum(log_actions, axis=-1, keepdims=True)
            # log_actions = self.policy.log_prob(self.states_ph, self.actions_ph)
        elif self.policy_type == "DiscretePolicy":
            action_probs = self.policy.model(self.states_ph)
            entropy = -tf_v1.reduce_sum(tf_v1.multiply(tf_v1.log(action_probs), action_probs), axis=-1)
            hot_encoded = tf_v1.one_hot(self.actions_ph, self.action_size)
            log_actions = tf_v1.log(tf_v1.reduce_sum(hot_encoded*action_probs, axis=-1))
        else:
            raise NotImplementedError(f"Received {self.policy_type}. This should have never happened!")

        log_loss = - log_actions*self.targets_ph
        entropy_loss = - self.alpha * entropy
        loss = log_loss + entropy_loss
        optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.policy.lr_ph)
        train_op = get_clipped_train_op(loss, optimizer, var_list=self.policy.trainable_vars, clip_norm=self.policy.clip_norm)
        self.policy.setup_loss(loss, train_op)

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
        feed_dict = {
            self.policy.model.input: states,
            self.states_ph: states,               # TODO: Remove default placeholders and define it in the algo class instead?
            self.actions_ph: actions,
            self.targets_ph: discounted_return
        }
        for _ in range(self.num_gradient_steps):
            self.policy.update(self.sess, feed_dict=feed_dict)

    def hook_before_train(self, **kwargs):
        self.init_actor_loss()
        super().hook_before_train(**kwargs)

    def hook_after_step(self, **kwargs):
        super().hook_after_step(**kwargs)
        state, action, reward, *_ = self.transition
        self.add_transition(state=state, action=action, reward=reward)
    

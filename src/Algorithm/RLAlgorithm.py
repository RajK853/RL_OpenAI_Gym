import numpy as np
from time import time
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from src.Utils import get_space_size, dict2str


class RLAlgorithm:

    def __init__(self, *, sess, env, policy, replay_buffer, batch_size, summary_dir, render, training=True,
                 goal_trials, goal_reward, display_interval, **kwargs):
        self.env = env
        self.policy = policy
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.action_size = get_space_size(self.action_space)
        self.observation_size = get_space_size(self.observation_space)
        self.sess = sess
        self.render = render
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self.display_interval = display_interval
        # Setup summaries
        self.summary_dir = summary_dir
        if summary_dir is None:
            self.summary_writer = None
        else:
            self.summary_writer = tf_v1.summary.FileWriter(summary_dir, sess.graph)
        self.summaries = self.setup_summaries()
        self.epoch = 0
        self.steps = 0
        self._saver = None
        self.training = training
        # Goal variables
        self.goal_trials = goal_trials
        self.goal_reward = goal_reward
        self.epoch_rewards = []
        self.goals_achieved = 0
        self.first_goal = (None, None)
        self.max_mean_reward = (-1, -1000)
        # Epoch variables
        self.epoch_reward = 0
        self.transition = None
        self.epoch_info = {}
        self._estimator_losses = []
        self._policy_losses = []

    @property
    def obs_shape(self):
        return self.observation_space.shape

    @property
    def action_shape(self):
        return self.action_space.shape

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf_v1.train.Saver(max_to_keep=10)
        return self._saver

    # TODO: How to handle different metrics of different environment? Max, mean and min of all?
    def setup_summaries(self):
        """
        Setup the summary placeholders and their scalars Tensors
        args:
            None
        returns:
            Tensor : Tensor with all summary scalars merged into
        """
        summaries = []
        with tf.name_scope("Performance_Summary_{}".format(self.env.spec.id)):
            self.epoch_reward_ph = tf_v1.placeholder(tf.float32, name="epoch_reward_ph")
            self.epoch_estimator_avg_loss_ph = tf_v1.placeholder(tf.float32, name="epoch_estimator_avg_loss_ph")
            self.epoch_policy_avg_loss_ph = tf_v1.placeholder(tf.float32, name="epoch_policy_avg_loss_ph")
            self.epoch_eps_ph = tf_v1.placeholder(tf.float32, name="epoch_eps_ph")
            epoch_reward_summary = tf_v1.summary.scalar("epoch_reward", self.epoch_reward_ph)
            epoch_estimator_avg_loss_summary = tf_v1.summary.scalar("epoch_estimator_avg_loss",
                                                                    self.epoch_estimator_avg_loss_ph)
            epoch_policy_avg_loss_summary = tf_v1.summary.scalar("epoch_policy_avg_loss", self.epoch_policy_avg_loss_ph)
            epoch_eps_summary = tf_v1.summary.scalar("epoch_eps", self.epoch_eps_ph)
            # TODO: Get summaries from algorithm, policy and env
            summaries.extend([epoch_estimator_avg_loss_summary, epoch_policy_avg_loss_summary, epoch_reward_summary,
                              epoch_eps_summary])
        return tf_v1.summary.merge(summaries)

    def get_summaries(self, feed_dict):
        """
        Executes summary op
        args:
            feed_dict (dict) : Feed dictionary as {self.epoch_reward_ph:epoch_reward,..., self.epoch_eps_ph:epoch_eps}
        returns:
            tf_v1.Summary: Summary protocol buffer
        """
        summaries = self.sess.run(self.summaries, feed_dict=feed_dict)
        return summaries

    def action(self, sess, states, **kwargs):
        if states.shape == self.obs_shape:
            states = states.reshape(1, *self.obs_shape)
        return self.policy.action(sess, states, **kwargs)

    def _run_once(self):
        """
        Runs the simulation for one epoch
        args:
            epoch (int) : Epoch index
            target_update_steps (int) : Global step interval to update weights of target estimator
            explore_ratio (float) : Fraction of explore_exploit_interval used to explore
            explore_exploit_interval (int) : Epoch interval to explore and exploit
            training (bool) : Training or testing the model
        returns:
            (float, float, float) : Mean loss, total reward and maximum position of the current epoch
        """
        self.hook_before_epoch()
        done = False
        state = self.env.reset()
        while not done:
            self.hook_before_step()
            action = self.action(self.sess, state, training=self.training)[0]
            next_state, reward, done, info = self.env.step(action)
            self.transition = (state, action, reward, next_state, done)
            if self.render:
                self.env.render()
            self.epoch_reward += reward
            state = next_state
            self.steps += 1
            self.hook_after_step()
        self.hook_at_epoch_end()

    def run(self, epochs):
        """
        Runs the simulation for several epochs
        args:
            epochs (int) : Total number of epochs
        returns:
            None
        """
        self.hook_before_train(epochs=epochs)
        mode_string = f"{'Training' if self.training else 'Testing'} agent. Please be patient..."
        print(mode_string, end="\r")
        t0 = time()
        for self.epoch in range(1, epochs + 1):
            self._run_once()
            # TODO: Make it not hardcoded later
            estimator_mean_loss = self.epoch_info["estimator_mean_loss"]
            policy_mean_loss = self.epoch_info["policy_mean_loss"]
            epoch_eps = self.epoch_info["eps"]
            self.hook_after_epoch()
            if not self.epoch % self.display_interval:  # Display epoch information
                t = time() - t0  # Measure time difference from previous display
                print(f"Epoch: {self.epoch}, mean_losses: {estimator_mean_loss:.4f}, {policy_mean_loss:.4f}, "
                      f"total_reward: {self.epoch_reward}, in {t:.4f} secs")
                print(mode_string, end="\r")
                t0 = time()
            # Log summaries to display in Tensorboard
            if self.summary_writer is not None:
                # TODO: Add goal information to the summary like goal achieved, goal reward?
                # TODO: Preparing feed_dict for the summary
                feed_dict = {self.epoch_reward_ph: self.epoch_reward,
                             self.epoch_estimator_avg_loss_ph: estimator_mean_loss,
                             self.epoch_policy_avg_loss_ph: policy_mean_loss, self.epoch_eps_ph: epoch_eps}
                summary = self.get_summaries(feed_dict)
                self.summary_writer.add_summary(summary, self.epoch)
        print("{:<50}".format(""), end="\r")
        self.hook_after_train()

    def save_model(self, chkpt_dir):
        """
        Saves variables from the session
        args:
            chkpt_dir (str) : Destination directory to store variables (as checkpoint)
        """
        self.saver.save(self.sess, chkpt_dir)

    def restore_model(self, chkpt_dir):
        """
        Restores varibles to the session
        args:
            chkpt_dir (str) : Source directory to restore variables (as checkpoint)
        """
        self.saver.restore(self.sess, chkpt_dir)

    @staticmethod
    def log(logger, model_name, parameter_dict, goal_summary):
        """
        Logs parameters and goal summary
        args:
            logger (logging.Logger) : Logger object
            model_name (str) : Name of model
            parameter_dict (dict) : Dictionary with parameters to log
            goal_summary (tuple) : Tuple with goal summary
        """
        parameter_str = dict2str(parameter_dict)
        logger.debug(f"{model_name} - {parameter_str}")
        num_goals, first_goal, max_goal = goal_summary
        first_goal_epoch, first_goal_reward = first_goal
        max_goal_epoch, max_goal_reward = max_goal
        logger.info(f"Goals achieved: {num_goals:<20}")
        if num_goals:
            logger.info(
                    f"First goal achieved: {first_goal_reward:.2f} mean reward at {first_goal_epoch} epoch.")
        logger.info(f"Max goal achieved: {max_goal_reward:.2f} mean reward at {max_goal_epoch} epoch.\n")

    def _init_diagnostics(self):
        pass

    def get_diagnostics(self):
        pass

    def hook_before_train(self, **kwargs):
        assert self.goal_trials <= kwargs["epochs"], "Number of epochs must be at least the number of goal trials!"
        print(f"Goal: Get average reward of {self.goal_reward:.2f} over {self.goal_trials} consecutive trials!")

    def hook_before_epoch(self, **kwargs):
        self.epoch_reward = 0.0

    def hook_before_step(self, **kwargs):
        pass

    def hook_after_step(self, **kwargs):
        pass

    def hook_at_epoch_end(self, **kwargs):
        self.epoch_rewards.append(self.epoch_reward)
        estimator_mean_loss = np.mean(self._estimator_losses) if self._estimator_losses else 0.0
        policy_mean_loss = np.mean(self._policy_losses) if self._policy_losses else 0.0
        policy_diagnostic = self.policy.get_diagnostic()
        self.epoch_info.update({"estimator_mean_loss": estimator_mean_loss,
                                "policy_mean_loss": policy_mean_loss},
                               **policy_diagnostic)

    def hook_after_epoch(self, **kwargs):
        if len(self.epoch_rewards) >= self.goal_trials:
            mean_reward = np.mean(self.epoch_rewards[-self.goal_trials:])
            solved = (mean_reward >= self.goal_reward)
            if solved:
                if self.first_goal[0] is None:
                     self.first_goal = (self.epoch, mean_reward)
                self.goals_achieved += 1
            if mean_reward > self.max_mean_reward[1]:
                self.max_mean_reward = (self.epoch, mean_reward)

    def hook_after_train(self, **kwargs):
        print(f"############# Goal Summary ############\nNumber of achieved goals: {self.goals_achieved}")
        if self.goals_achieved:
            print(f"First goal achieved at epoch {self.first_goal[0]} with reward {self.first_goal[1]}")
        trials = self.goal_trials
        epoch, reward = self.max_mean_reward
        print(f"Max mean reward over {trials} trials achieved at epoch {epoch} with reward {reward}")

import numpy as np
from time import perf_counter
import tensorflow.compat.v1 as tf_v1
from collections import deque

from src.progressbar import ProgressBar
from src.utils import get_space_size, dict2str


class RLAlgorithm:
    VALID_POLICIES = []

    def __init__(self, *, sess, env, policy, replay_buffer, batch_size, render, summary_dir=None, training=True,
                 goal_trials, goal_reward, display_interval):
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
        self.tag = f"{self.env.spec.id}"
        self.summary_dir = summary_dir
        if summary_dir is None:
            self.summary_writer = None
        else:
            self.summary_writer = tf_v1.summary.FileWriter(summary_dir, sess.graph)
        self._saver = None
        self.training = training
        # Goal variables
        self.goal_trials = goal_trials
        self.goal_reward = goal_reward
        self.epoch_rewards = []
        self.goals_achieved = 0
        self.first_goal = (None, None)
        self.max_mean_reward = (-1, -1000)
        self.mean_reward = 0
        # Epoch variables
        self.steps = 0
        self.epoch = 0
        self.epoch_length = 0
        self.epoch_reward = 0
        self.transition = None
        self.process_action = None
        self.summary_init_objects = (self.policy, )
        self.scalar_summaries = ("epoch_reward", "epoch_length")
        self.histogram_summaries = ()

    def validate_policy(self):
        policy_type = self.policy.__class__.__name__
        algo_type = self.__class__.__name__
        assert len(self.VALID_POLICIES) == 0 or policy_type in self.VALID_POLICIES, \
            f"{algo_type} only supports '{self.VALID_POLICIES}' and not {policy_type}!"

    def init_summaries(self):
        for obj in self.summary_init_objects:
            obj.init_summaries(tag=self.tag)

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

    def action(self, sess, states, **kwargs):
        return self.policy.action(sess, states, **kwargs)

    def step(self, state):
        self.hook_before_step()
        raw_action = self.action(self.sess, state, training=self.training)
        action = self.process_action(raw_action)
        next_state, reward, done, info = self.env.step(action)
        self.transition = (state, action, reward, next_state, int(done))
        self.epoch_reward += reward
        self.hook_after_step()
        return next_state, done

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
        done = 0
        state = self.env.reset()
        while not done:
            state, done = self.step(state)
            if self.render:
                self.env.render()
        self.hook_at_epoch_end()

    def run(self, total_epochs):
        """
        Runs the simulation for several epochs
        args:
            total_epochs (int) : Total number of epochs
        """

        self.hook_before_train(epochs=total_epochs)
        mode_string = f"  {'Training' if self.training else 'Testing'} agent:"
        pbar = ProgressBar(total_iter=total_epochs, display_text=mode_string)
        timing_queue = deque(maxlen=50)
        start_time = perf_counter()
        for self.epoch in range(1, total_epochs + 1):
            t0 = perf_counter()
            self._run_once()
            self.hook_after_epoch()
            t1 = perf_counter()
            dt = t1-t0
            timing_queue.append(dt)
            mean_time_per_epoch = np.mean(timing_queue)
            time_left_sec = (total_epochs - self.epoch)*mean_time_per_epoch
            time_left_minute = time_left_sec/60
            elapsed_time_sec = t1-start_time
            elapsed_time_minute = elapsed_time_sec/60
            time_info_text = f"[Mean time per epoch: {mean_time_per_epoch:3.2f} seconds, " \
                             f"Elapsed time: {elapsed_time_minute:3.2f} minutes, ETA: {time_left_minute:3.2f} minutes]"
            pbar.step(add_text=time_info_text)
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
        logger.info(f"  Goals achieved: {num_goals:<20}")
        if num_goals:
            logger.info(f"  First goal achieved: {first_goal_reward:.3f} mean reward at {first_goal_epoch} epoch.")
        logger.info(f"  Max goal achieved: {max_goal_reward:.3f} mean reward at {max_goal_epoch} epoch.\n")

    def write_summary(self, name, **kwargs):
        summary = tf_v1.Summary(value=[tf_v1.Summary.Value(tag=name, **kwargs)])
        self.summary_writer.add_summary(summary, self.epoch)

    def add_summaries(self):
        def get_histogram(values):
            counts, bin_edges = np.histogram(values)
            # Fill fields of histogram proto
            hist = tf_v1.HistogramProto()
            hist.min = float(np.min(values))
            hist.max = float(np.max(values))
            hist.num = int(np.prod(values.shape))
            hist.sum = float(np.sum(values))
            hist.sum_squares = float(np.sum(values ** 2))
            bin_edges = bin_edges[1:]
            hist.bucket_limit = bin_edges
            hist.bucket = counts
            return hist

        if self.summary_writer is not None:
            for summary_attr in self.scalar_summaries:
                attr = getattr(self, summary_attr)
                self.write_summary(f"{self.tag}/{summary_attr}", simple_value=attr)
            for summary_attr in self.histogram_summaries:
                attr = np.array(getattr(self, summary_attr))
                hist = get_histogram(attr)
                self.write_summary(f"{self.tag}/{summary_attr}", histo=hist)
            for obj in self.summary_init_objects:
                summary = getattr(obj, "summary")
                if summary:
                    self.summary_writer.add_summary(summary, self.epoch)

    def hook_before_train(self, **kwargs):
        assert self.goal_trials <= kwargs["epochs"], "Number of epochs must be at least the number of goal trials!"
        print(f"\n# Goal: Get average reward of {self.goal_reward:.3f} over {self.goal_trials} consecutive trials!")
        sample_action = self.env.action_space.sample()
        if isinstance(sample_action, np.ndarray):
            self.process_action = lambda a: np.array([np.squeeze(a)]) if len(sample_action) == 1 else np.squeeze(a)
        else:
            self.process_action = lambda a: np.squeeze(a).item()
        self.init_summaries()

    def hook_before_epoch(self, **kwargs):
        self.epoch_reward = 0.0
        self.epoch_length = 0

    def hook_before_step(self, **kwargs):
        self.steps += 1
        self.epoch_length += 1

    def hook_after_step(self, **kwargs):
        pass

    def hook_at_epoch_end(self, **kwargs):
        self.epoch_rewards.append(self.epoch_reward)

    def hook_after_epoch(self, **kwargs):
        if len(self.epoch_rewards) >= self.goal_trials:
            self.mean_reward = np.mean(self.epoch_rewards[-self.goal_trials:])
            if self.mean_reward >= self.goal_reward:
                if self.first_goal[0] is None:
                    self.first_goal = (self.epoch, self.mean_reward)
                self.goals_achieved += 1
            if self.mean_reward > self.max_mean_reward[1]:
                self.max_mean_reward = (self.epoch, self.mean_reward)

    def hook_after_train(self, **kwargs):
        print(f"\n# Goal Summary")
        print(f"  Number of achieved goals: {self.goals_achieved}")
        if self.goals_achieved:
            print(f"  First goal achieved at epoch {self.first_goal[0]} with reward {self.first_goal[1]:.3f}")
        trials = self.goal_trials
        epoch, reward = self.max_mean_reward
        print(f"  Best mean reward over {trials} trials achieved at epoch {epoch} with reward {reward:.3f}")

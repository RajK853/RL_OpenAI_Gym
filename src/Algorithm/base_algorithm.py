import numpy as np
import os.path as os_path
import tensorflow.compat.v1 as tf_v1

from src.progressbar import ProgressBar
from src.utils import get_space_size, json_dump


class BaseAlgorithm:
    VALID_POLICIES = {}

    def __init__(self, *, sess, env, policy, batch_size, render, goal_trials, goal_reward, load_model=None,
                 summary_dir=None, training=True):
        self.env = env
        self.policy = policy
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.action_size = get_space_size(self.action_space)
        self.observation_size = get_space_size(self.observation_space)
        self.sess = sess
        self.render = render
        self.batch_size = batch_size
        self.load_model = load_model
        # Setup summaries
        self.tag = self.env.spec.id
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
        self.summary_init_objects = (self.policy, )
        self.scalar_summaries = ("epoch_reward", "epoch_length")
        self.histogram_summaries = ()
        self.validate_policy()

    def validate_policy(self):
        policy_type = self.policy.__class__.__name__
        algo_type = self.__class__.__name__
        assert len(self.VALID_POLICIES) == 0 or policy_type in self.VALID_POLICIES, \
            f"{algo_type} only supports '{self.VALID_POLICIES}' and not {policy_type}!"

    @staticmethod
    def get_summaries(obj):
        return obj.scalar_summaries + obj.histogram_summaries

    def init_summaries(self):
        summaries = {"algo": self.get_summaries(self)}
        for obj in self.summary_init_objects:
            obj.init_summaries(tag=self.tag)
            summaries[obj.scope] = self.get_summaries(obj)
        # summaries = dict2str(summaries)
        print("\n# Initialized summaries:")
        for key, value in summaries.items():
            print(f"  {key}: {value}")

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

    def action(self, states, **kwargs):
        action = self.policy.action(self.sess, states, **kwargs)
        return action

    def _step(self, env, state):
        action = self.action(state)[0]
        # print(action)
        next_state, reward, done, info = env.step(action)
        self.transition = (state, action, reward, next_state, int(done))
        self.epoch_reward += reward
        return next_state, done

    def step(self, state):
        self.hook_before_step()
        next_state, done = self._step(self.env, state)
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
        progressbar = ProgressBar(total_epochs, title=f"# {'Training' if self.training else 'Testing'} agent:", info_text="Epoch: ({epoch}/%s)"%(total_epochs))
        print()
        for self.epoch in range(1, total_epochs + 1):
            self._run_once()
            self.hook_after_epoch()
            progressbar.step(epoch=self.epoch)
        self.hook_after_train()

    def save_model(self, chkpt_dir):
        """
        Saves variables from the session
        args:
            chkpt_dir (str) : Destination directory to store variables (as checkpoint)
        """
        self.saver.save(self.sess, chkpt_dir)
        print(f"# Saved model weights to '{chkpt_dir}'")

    def restore_model(self, chkpt_dir):
        """
        Restores variables to the session
        args:
            chkpt_dir (str) : Source directory to restore variables (as checkpoint)
        """
        self.saver.restore(self.sess, chkpt_dir)
        print(f"# Restored model weights from '{chkpt_dir}'")

    def write_summary(self, name, step, **kwargs):
        summary = tf_v1.Summary(value=[tf_v1.Summary.Value(tag=name, **kwargs)])
        self.summary_writer.add_summary(summary, step)

    def add_summaries(self, step):
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
                self.write_summary(f"{self.tag}/{summary_attr}", step=step, simple_value=attr)
            for summary_attr in self.histogram_summaries:
                attr = np.array(getattr(self, summary_attr))
                histogram = get_histogram(attr)
                self.write_summary(f"{self.tag}/{summary_attr}", step=step, histo=histogram)
            for obj in self.summary_init_objects:
                summary = getattr(obj, "summary")
                if summary:
                    self.summary_writer.add_summary(summary, step)

    def hook_before_train(self, **kwargs):
        if self.training:
            self.init_summaries()
            # assert self.goal_trials <= kwargs["epochs"], "Number of epochs must be at least the number of goal trials!"
            print(f"\n# Goal: Get average reward of {self.goal_reward:.1f} over {self.goal_trials} consecutive trials!")
            self.sess.run(tf_v1.global_variables_initializer())
        else:
            assert self.load_model is not None, "No model given to test!"
        if self.load_model is not None:
            self.restore_model(os_path.join(self.load_model, "model.chkpt"))          # Restore model variables

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
        if self.training:
            print(f"\n# Goal Summary")
            print(f"  Number of achieved goals: {self.goals_achieved}")
            if self.goals_achieved:
                print(f"  First goal achieved at epoch {self.first_goal[0]} with reward {self.first_goal[1]:.3f}")
            trials = self.goal_trials
            epoch, reward = self.max_mean_reward
            print(f"  Best mean reward over {trials} trials achieved at epoch {epoch} with reward {reward:.3f}")
            json_file = os_path.join(os_path.dirname(self.summary_dir), "goal_info.json")
            dump_dict = {"num_goals_achieved": self.goals_achieved,
                         "first_goal": dict(zip(("epoch", "reward"), self.first_goal)),
                         "max_mean_reward": dict(zip(("epoch", "reward"), self.max_mean_reward))}
            json_dump(dump_dict, file_name=json_file, indent=4)
            print(f"\n# Goal info saved at '{json_file}'")

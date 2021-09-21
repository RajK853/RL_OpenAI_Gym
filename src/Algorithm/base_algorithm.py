import tqdm
import numpy as np
import os.path as os_path
import tensorflow.compat.v1 as tf_v1

from src.utils import get_space_size, json_dump, dump_yaml


def get_max_episode_steps(env):
    value = getattr(env, "_max_episode_steps", None)
    if value is not None:
        return value
    if hasattr(env, "env"):
        return get_max_episode_steps(env.env)
    raise


class BaseAlgorithm:
    VALID_POLICIES = {}
    PARAMETERS = {
        "render", "goal_trials", "goal_reward", "load_model", "layers", "preprocessors", "clip_norm",
        "seed", "num_gradient_steps", "max_episode_steps"
    }

    def __init__(self, *, sess, env, policy, render, goal_trials, goal_reward, load_model=None, summary_dir=None, 
        layers=None, preprocessors=None, clip_norm=None, seed=None, num_gradient_steps="auto", max_episode_steps="auto", 
        name="algo"):
        self.seed = seed
        self.scope = name
        self.env = env
        self.policy = policy
        self._layers = layers
        self.preprocessors = preprocessors
        self.algo_type = self.__class__.__name__
        self.policy_type = self.policy.__class__.__name__
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.action_size = get_space_size(self.action_space)
        self.observation_size = get_space_size(self.observation_space)
        self.max_episode_steps = get_max_episode_steps(self.env) if max_episode_steps == "auto" else max_episode_steps
        self.num_gradient_steps = self.max_episode_steps if num_gradient_steps == "auto" else num_gradient_steps
        self.sess = sess
        self.render = render
        self.clip_norm = clip_norm
        self.load_model = load_model
        # Setup summaries
        self.tag = self.env.spec.id
        self.summary_dir = summary_dir
        self.summary_writer = None
        self._saver = None
        # Goal variables
        self.goal_trials = goal_trials
        self.goal_reward = goal_reward
        self.epoch_rewards = []
        self.goals_achieved = 0
        self.first_goal = (None, None)
        self.max_mean_reward = (-1, -1000)
        self.mean_reward = 0
        self.best_epoch_reward = None
        # Epoch variables
        self.steps = 0
        self.epoch = 0
        self.epoch_length = 0
        self.epoch_reward = 0
        self.transition = None
        self.schedulers = ()
        self.summary_init_objects = (self.policy, )
        self.scalar_summaries = ("epoch_reward", "epoch_length")
        self.histogram_summaries = ()
        self.scalar_summaries_tf = ()
        self.histogram_summaries_tf = ()
        self.validate_policy()

    def get_params(self):
        return {attr_name: getattr(self, attr_name) for attr_name in self.PARAMETERS}

    def validate_policy(self):
        assert len(self.VALID_POLICIES) == 0 or self.policy_type in self.VALID_POLICIES, \
            f"{self.algo_type} only supports '{self.VALID_POLICIES}' and not {self.policy_type}!"

    def increment_schedulers(self):
        for scheduler in self.schedulers:
            scheduler.increment()

    @staticmethod
    def get_summaries(obj):
        return obj.scalar_summaries + obj.scalar_summaries_tf + obj.histogram_summaries + obj.histogram_summaries_tf

    def init_summaries(self):
        summaries = {"algo": self.get_summaries(self)}
        for obj in self.summary_init_objects:
            obj.init_summaries(tag=self.tag)
            summaries[obj.scope] = self.get_summaries(obj)
        print("\n# Initialized summaries:")
        for key, value in summaries.items():
            print(f"  {key}: {value}")

    @property
    def layers(self):
        return self._layers

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
        next_state, reward, done, info = env.step(action)
        self.transition = [state, action, reward, next_state, int(done)]
        self.epoch_reward += reward
        return next_state, done

    def step(self, state):
        self.hook_before_step()
        next_state, done = self._step(self.env, state)
        self.hook_after_step()
        return next_state, done

    def dump_goal_summary(self):
        print(f"\n# Goal Summary")
        print(f"  Number of achieved goals: {self.goals_achieved}")
        if self.goals_achieved:
            print(f"  First goal achieved at epoch {self.first_goal[0]} with reward {self.first_goal[1]:.3f}")
        epoch, reward = self.max_mean_reward
        print(f"  Best mean reward over {self.goal_trials} trials achieved at epoch {epoch} with reward {reward:.3f}")

        file_path = os_path.join(os_path.dirname(self.summary_dir), "goal_info.yaml")
        dump_dict = {"num_goals_achieved": self.goals_achieved,
                     "first_goal": dict(zip(("epoch", "reward"), self.first_goal)),
                     "max_mean_reward": dict(zip(("epoch", "reward"), self.max_mean_reward))}
        if file_path.endswith(".json"):
            json_dump(dump_dict, file_name=file_path, indent=4)
        elif file_path.endswith(".yaml"):
            dump_yaml(dump_dict, file_path=file_path)
        print(f"\n# Goal info saved at '{file_path}'")


    def _run_once(self):
        """
        Runs the simulation for one epoch
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
        for self.epoch in tqdm.trange(1, total_epochs + 1, desc="# Training agent"):
            self._run_once()
            self.hook_after_epoch()
        self.hook_after_train()

    def write_summary(self, name, step, **kwargs):
        summary = tf_v1.Summary(value=[tf_v1.Summary.Value(tag=name, **kwargs)])
        self.summary_writer.add_summary(summary, step)

    def add_summaries(self, step):
        def get_histogram(values):
            hist = tf_v1.HistogramProto()
            # Fill attributes in the histogram Proto
            counts, bin_edges = np.histogram(values)
            hist.min = float(np.min(values))
            hist.max = float(np.max(values))
            hist.num = int(np.prod(values.shape))
            hist.sum = float(np.sum(values))
            hist.sum_squares = float(np.sum(values ** 2))
            hist.bucket_limit = bin_edges[1:]
            hist.bucket = counts
            return hist

        def write_non_tensor_summaries(obj):
            summary_scope = f"{self.tag}/{obj.scope}"
            for summary_attr in obj.scalar_summaries:
                attr = getattr(obj, summary_attr)
                self.write_summary(f"{summary_scope}/{summary_attr}", step=step, simple_value=attr)
            for summary_attr in obj.histogram_summaries:
                attr = np.array(getattr(obj, summary_attr))
                histogram = get_histogram(attr)
                self.write_summary(f"{summary_scope}/{summary_attr}", step=step, histo=histogram)

        def write_tensor_summaries(obj):
            if hasattr(summ_obj, "summary"):      # summ_obj.summary = summary of all gathered tensors of that object
                summary = getattr(summ_obj, "summary")
                if summary:
                    self.summary_writer.add_summary(summary, step)

        if self.summary_writer is not None:
            for summ_obj in [self, *self.summary_init_objects]:
                write_non_tensor_summaries(summ_obj)
                write_tensor_summaries(summ_obj)         

    def update_reward_info(self):        
        self.mean_reward = float(np.mean(self.epoch_rewards[-self.goal_trials:]))
        if self.mean_reward >= self.goal_reward:
            if self.first_goal[0] is None:
                self.first_goal = (self.epoch, self.mean_reward)
            self.goals_achieved += 1
        if self.mean_reward >= self.max_mean_reward[1]:
            self.max_mean_reward = (self.epoch, self.mean_reward)

    def save_model(self, chkpt_dir, verbose=True):
        """
        Saves variables from the session
        :param chkpt_dir: (str) Destination directory to store variables (as checkpoint)
        """
        self.saver.save(self.sess, chkpt_dir)
        if verbose:
            print(f"# Saved model weights to '{chkpt_dir}'")

    def restore_model(self, chkpt_dir, verbose=True):
        """
        Restores variables to the session
        :param chkpt_dir: (str) Source directory to restore variables (as checkpoint)
        """
        self.saver.restore(self.sess, chkpt_dir)
        if verbose:
            print(f"# Restored model weights from '{chkpt_dir}'")

    def hook_before_train(self, **kwargs):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.policy.hook_before_train(**kwargs)
        self.init_summaries()
        print(f"\n# Goal: Get average reward of {self.goal_reward:.1f} over {self.goal_trials} consecutive trials!")
        self.sess.run(tf_v1.global_variables_initializer())
        if self.load_model is not None:
            self.restore_model(os_path.join(self.load_model, "model.chkpt"))          # Restore model variables
        if self.summary_dir is not None:
            self.summary_writer = tf_v1.summary.FileWriter(self.summary_dir, self.sess.graph)
        self.steps = 0

    def hook_before_epoch(self, **kwargs):
        self.epoch_reward = 0.0
        self.epoch_length = 0
        self.policy.hook_before_epoch(**kwargs)

    def hook_before_step(self, **kwargs):
        self.steps += 1
        self.epoch_length += 1
        self.policy.hook_before_step(**kwargs)

    def hook_after_step(self, **kwargs):
        self.policy.hook_after_step(**kwargs)

    def hook_at_epoch_end(self, **kwargs):
        self.epoch_rewards.append(self.epoch_reward)
        if self.best_epoch_reward is None or (self.epoch_reward >= self.best_epoch_reward):
            self.best_epoch_reward = self.epoch_reward
            chkpt_path = os_path.join(self.summary_dir, "model.chkpt")
            self.save_model(chkpt_path, verbose=False)
            dump_path = os_path.join(os_path.dirname(self.summary_dir), "policy.tf")
            self.policy.save(dump_path, verbose=False)

    def hook_after_epoch(self, **kwargs):
        self.policy.hook_after_epoch(**kwargs)
        self.train()
        self.increment_schedulers()
        if len(self.epoch_rewards) >= self.goal_trials:
            self.update_reward_info()
        self.add_summaries(self.epoch)

    def hook_after_train(self, **kwargs):
        self.policy.hook_after_train(**kwargs)
        self.dump_goal_summary()

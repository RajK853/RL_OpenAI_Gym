import tqdm

from .base_algorithm import BaseAlgorithm
from src.Buffer import ReplayBuffer
from src.utils import get_scheduler

from numpy import random

DEFAULT_KWARGS = {
    "batch_size_kwargs": {
        "type": "ConstantScheduler",
        "value":  100,
    },
}


class OffPolicyAlgorithm(BaseAlgorithm):
    PARAMETERS = BaseAlgorithm.PARAMETERS.union({"batch_size_kwargs", "num_init_exp_samples", "max_init_exp_timestep", "buffer_size"})

    def __init__(self, *, batch_size_kwargs=DEFAULT_KWARGS["batch_size_kwargs"], num_init_exp_samples=10000, max_init_exp_timestep="auto", 
        buffer_size=1_000_000, **kwargs):
        super(OffPolicyAlgorithm, self).__init__(**kwargs)
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(size=buffer_size)
        self.batch_size_kwargs = batch_size_kwargs
        self.batch_size_schedhuler = get_scheduler(batch_size_kwargs)
        self.schedulers += (self.batch_size_schedhuler, )
        self.num_init_exp_samples = None if num_init_exp_samples is None else int(num_init_exp_samples)
        self.max_init_exp_timestep = self.max_episode_steps if max_init_exp_timestep == "auto" else max_init_exp_timestep
        self.scalar_summaries += ("replay_buffer_size", "batch_size")

    @property
    def batch_size(self):
        return int(self.batch_size_schedhuler.value)

    @property
    def replay_buffer_size(self):
        return len(self.replay_buffer)

    def init_explore(self, env, total_steps):
        print()
        steps = 0
        pbar = tqdm.tqdm(total=total_steps, desc="# Collecting initial samples")
        while steps < total_steps:
            done = False
            state = env.reset()
            epoch_steps = 0
            while not done:
                action = self.action(state)[0]  # env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                self.transition = [state, action, reward, next_state, int(done)]
                self.replay_buffer.add(self.transition)    # TODO: Any problem in using hook_after_step?
                epoch_steps += 1
                pbar.update()
                if ((steps + epoch_steps) >= total_steps) or (epoch_steps >= self.max_init_exp_timestep):
                    break
            steps += epoch_steps
            # pbar.refresh()
        env.reset()

    def hook_after_step(self, **kwargs):
        super().hook_after_step(**kwargs)
        self.replay_buffer.add(self.transition)

    def hook_before_train(self, **kwargs):
        super(OffPolicyAlgorithm, self).hook_before_train(**kwargs)
        if self.num_init_exp_samples is not None:
            # Explore in the original environment which does not record video during rollout
            self.init_explore(env=self.env.env, total_steps=self.num_init_exp_samples)

    def train(self):
        raise NotImplementedError

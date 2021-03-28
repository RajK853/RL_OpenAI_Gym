from src.Buffer import ReplayBuffer
from .base_algorithm import BaseAlgorithm
from ..progressbar import ProgressBar


class OffPolicyAlgorithm(BaseAlgorithm):

    def __init__(self, *, num_init_exp_samples=None, buffer_size=800_000, **kwargs):
        super(OffPolicyAlgorithm, self).__init__(**kwargs)
        self.num_init_exp_samples = int(num_init_exp_samples) if num_init_exp_samples is not None else num_init_exp_samples  # Number of initial exploration samples
        self.replay_buffer = ReplayBuffer(size=buffer_size)
        self.scalar_summaries += ("buffer_size", )

    def init_explore(self):
        # Unwrap the original environment which does not record video
        env = self.env.unwrapped
        total_steps = self.num_init_exp_samples
        pbar = ProgressBar(total_steps, title=f"# Collecting {total_steps} initial samples:", display_interval=100)
        steps = 0
        if self.load_model is None:
            action_sample_func = lambda s: env.action_space.sample()  
        else:
            action_sample_func = lambda s: self.action(s)[0]
        print()
        while steps < total_steps:
            done = False
            state = env.reset()
            while not done and (steps < total_steps):
                action = action_sample_func(state)
                next_state, reward, done, info = env.step(action)
                self.transition = (state, action, reward, next_state, int(done))
                self.replay_buffer.add(self.transition)
                steps += 1
                pbar.step()
        env.reset()

    def hook_before_train(self, **kwargs):
        super(OffPolicyAlgorithm, self).hook_before_train(**kwargs)
        if self.training:
            if self.num_init_exp_samples is not None:
                self.init_explore()

    @property
    def buffer_size(self):
        return len(self.replay_buffer)

    def train(self):
        raise NotImplementedError

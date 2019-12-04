import copy
import tensorflow as tf

from src.Policy import GreedyEpsilonPolicy, GaussianPolicy, UniformPolicy
from src.Algorithm import DQN, DDQN, PolicyGradient
from src.Buffer import ReplayBuffer, PrioritizedReplayBuffer

ENV_NAMES = ("CartPole-v0", "LunarLander-v2", "MountainCar-v0")

BASE_CONFIG = {
    "batch_size": 100,
    "buffer": "replay_buffer",
}

_ReplayBuffer = {
    "function": ReplayBuffer,
    "kwargs": {
        "size": 50000,
    }
}

_PrioritizedReplayBuffer = {
    "function": PrioritizedReplayBuffer,
    "kwargs": {
        "size": 50000,
    }
}

REPLAY_BUFFERS = {
    "replay_buffer" : _ReplayBuffer,
    "prioritized_replay_buffer": _PrioritizedReplayBuffer,
}

# Policies
GreedyEpsilon = {
    "function": GreedyEpsilonPolicy,
    "kwargs": {
        "eps_range": (1, 0.001),
        "eps_decay": 0.87,
        "explore_ratio": 0.80,
        "explore_exploit_interval": 20,
    }
}

Gaussian = {
    "function": GaussianPolicy,
    "kwargs": {
        "layer_units": (20, 20),
    }
}

Uniform = {
    "function": UniformPolicy,
    "kwargs": {
        "layer_units": (20, 20),
        "activation": tf.nn.relu,
    }
}

POLICIES = {
    "greedy_epsilon": GreedyEpsilon,
    "gaussian": Gaussian,
    "uniform": Uniform,
}

# Algorithms
_DQN = {
    "function": DQN,
    "kwargs": {
        "lr": 0.989,
        "df": 0.996,
    }
}

_DDQN = {
    "function": DDQN,
    "kwargs": {
        **_DQN["kwargs"],
        "tau": 0.671,
        "update_interval": 24,
    }
}

_PolicyGradient = {
    "function": PolicyGradient,
    "kwargs": {
        "gamma": 0.7,
        "num_train": 2,
    }
}

ALGORITHMS = {
    "dqn": _DQN,
    "ddqn": _DDQN,
    "policy_gradient": _PolicyGradient,
}


def get_configuration(args):
    replay_buffer = copy.deepcopy(REPLAY_BUFFERS[args.buffer])
    policy = copy.deepcopy(POLICIES[args.policy])
    algorithm = copy.deepcopy(ALGORITHMS[args.algorithm])
    if args.algorithm in ("policy_gradient", ):
        assert args.policy in ("uniform", ), f"{args.algorithm} only supports uniform policy and not {args.policy}"
    algorithm["kwargs"].update(render=args.render,
                               goal_trials=args.goal_trials,
                               goal_reward=args.goal_reward,
                               display_interval=args.display_interval,
                               **copy.deepcopy(BASE_CONFIG))
    config_dict = {
        "buffer_param": replay_buffer,
        "policy_param": policy,
        "algorithm_param": algorithm,
    }
    return config_dict


def get_func_and_kwargs_from_param(param):
    return param["function"], param["kwargs"]


def get_buffer_from_variant(variant):
    param = variant["buffer_param"]
    func, kwargs = get_func_and_kwargs_from_param(param)
    return func(**kwargs)


def get_policy_from_variant(env, variant):
    param = variant["policy_param"]
    func, kwargs = get_func_and_kwargs_from_param(param)
    return func(env=env, **kwargs)


def get_algorithm_from_variant(sess, env, policy, buffer, summary_dir, variant):
    param = variant["algorithm_param"]
    func, kwargs = get_func_and_kwargs_from_param(param)
    return func(sess=sess, env=env, policy=policy, replay_buffer=buffer, summary_dir=summary_dir, **kwargs)
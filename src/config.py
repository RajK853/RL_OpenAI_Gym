import copy
import tensorflow as tf
from tensorflow.keras.regularizers import l2

from src.Policy import GreedyEpsilonPolicy, GaussianPolicy, DiscretePolicy
from src.Algorithm import DQN, DDQN, Reinforce, A2C
from src.Buffer import ReplayBuffer, PrioritizedReplayBuffer

ALGO_LEARNING_RATE = 1e-4
POLICY_LEARNING_RATE = 5e-4

BASE_CONFIG = {
    "batch_size": 100,
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
        "eps_decay": 0.01,
        "explore_ratio": 0.60,
        "explore_exploit_interval": 20,
    }
}

Gaussian = {
    "function": GaussianPolicy,
    "kwargs": {
        "alpha": 2e-4,
        "lr": POLICY_LEARNING_RATE,
        "layer_units": (50, 50),
        "activation": tf.nn.relu,
        "kernel_regularizer": l2(1e-6),
    }
}

Discrete = {
    "function": DiscretePolicy,
    "kwargs": {
        "lr": POLICY_LEARNING_RATE,
        "layer_units": (50, 50),
        "activation": tf.nn.relu,
        "kernel_regularizer": l2(1e-6),
    }
}

POLICIES = {
    "greedy_epsilon": GreedyEpsilon,
    "gaussian": Gaussian,
    "discrete": Discrete,
}

# Algorithms
_DQN = {
    "function": DQN,
    "kwargs": {
        "lr": 0.989,
        "df": 0.996,
        "layer_units": (50, 50),
    }
}

_DDQN = {
    "function": DDQN,
    "kwargs": {
        **_DQN["kwargs"],
        "tau": 0.01,
        "update_interval": 1,
    }
}

_Reinforce = {
    "function": Reinforce,
    "kwargs": {
        "gamma": 0.9,
        "num_train": 4,
    }
}

_A2C = {
    "function": A2C,
    "kwargs": {
        **_Reinforce["kwargs"]
    }
}

ALGORITHMS = {
    "dqn": _DQN,
    "ddqn": _DDQN,
    "reinforce": _Reinforce,
    "a2c": _A2C,
}


def get_configuration(args):
    replay_buffer = copy.deepcopy(REPLAY_BUFFERS[args.buffer])
    policy = copy.deepcopy(POLICIES[args.policy])
    algorithm = copy.deepcopy(ALGORITHMS[args.algorithm])
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


def get_algorithm_from_variant(sess, env, policy, buffer, summary_dir, training, variant):
    param = variant["algorithm_param"]
    func, kwargs = get_func_and_kwargs_from_param(param)
    return func(sess=sess, env=env, policy=policy, replay_buffer=buffer, summary_dir=summary_dir,
                training=training, **kwargs)
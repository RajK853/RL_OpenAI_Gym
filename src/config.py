import copy
import tensorflow.compat.v1 as tf_v1
from tensorflow.keras.regularizers import l2

from src.Policy import GreedyEpsilonPolicy, GaussianPolicy, DiscretePolicy, ContinuousPolicy
from src.Algorithm import DQN, DDQN, Reinforce, ActorCritic, Sarsa, DDPG
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
    "replay_buffer": _ReplayBuffer,
    "prioritized_replay_buffer": _PrioritizedReplayBuffer,
}

# Policies
GreedyEpsilon = {
    "function": GreedyEpsilonPolicy,
    "kwargs": {
        "eps_range": (0.9, 0.001),
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
        "layer_units": (100, 100),
        "layer_kwargs": {
            "activation": tf_v1.nn.relu,
            "kernel_regularizer": l2(1e-6),
        }
    }
}

Discrete = {
    "function": DiscretePolicy,
    "kwargs": {
        "lr": POLICY_LEARNING_RATE,
        "layer_units": (50, 50, 50),
        "layer_kwargs": {
            "activation": tf_v1.nn.relu,
            "kernel_regularizer": l2(1e-6),
        }
    }
}

Continuous = {
    "function": ContinuousPolicy,
    "kwargs": {
        "lr": POLICY_LEARNING_RATE,
        "layer_units": (100, 100, 50),
        "layer_kwargs": {
            "activation": tf_v1.nn.relu,
            "kernel_regularizer": l2(1e-6),
        },
        "output_kwargs": {
            "activation": tf_v1.nn.tanh,
        }
    }
}

POLICIES = {
    "greedy_epsilon": GreedyEpsilon,
    "gaussian": Gaussian,
    "discrete": Discrete,
    "continuous": Continuous,
}

# Algorithms
_DQN = {
    "function": DQN,
    "kwargs": {
        "lr": 0.989,
        "gamma": 0.996,
        "layer_units": (50, 50, 50),
    }
}

_Sarsa = {
    "function": Sarsa,
    "kwargs": {
        "lr": 0.99,
        "gamma": 0.996,
        "layer_units": (50, 50, 25),
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

_DDPG = {
    "function": DDPG,
    "kwargs": {
        "lr": 0.996,
        "gamma": 0.996,
        "layer_units": (100, 100, 50),
        "tau": 0.1,
        "update_interval": 1,
        "sigma": 0.01,           # Noise value = sigma * random.uniform(-1, 1)
    }
}

_Reinforce = {
    "function": Reinforce,
    "kwargs": {
        "gamma": 0.9,
        "num_train": 2,
    }
}

_ActorCritic = {
    "function": ActorCritic,
    "kwargs": {
        **_Reinforce["kwargs"]
    }
}

ALGORITHMS = {
    "dqn": _DQN,
    "ddqn": _DDQN,
    "reinforce": _Reinforce,
    "actor_critic": _ActorCritic,
    "sarsa": _Sarsa,
    "ddpg": _DDPG,
}


def get_configuration(args):
    replay_buffer = copy.deepcopy(REPLAY_BUFFERS[args.buffer])
    policy = copy.deepcopy(POLICIES[args.policy])
    algorithm = copy.deepcopy(ALGORITHMS[args.algorithm])
    algorithm["kwargs"].update(render=args.render,
                               **copy.deepcopy(BASE_CONFIG))
    config_dict = {
        # "buffer_param": replay_buffer,
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


def get_algorithm_from_variant(sess, env, policy, summary_dir, training, goal_trials, goal_reward, variant):
    param = variant["algorithm_param"]
    func, kwargs = get_func_and_kwargs_from_param(param)
    return func(sess=sess, env=env, policy=policy, summary_dir=summary_dir,
                training=training, goal_trials=goal_trials, goal_reward=goal_reward, **kwargs)
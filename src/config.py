import copy
import tensorflow.compat.v1 as tf_v1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm

from src.Policy import GreedyEpsilonPolicy, GaussianPolicy, DiscretePolicy, ContinuousPolicy
from src.Algorithm import DQN, DDQN, Reinforce, ActorCritic, Sarsa, DDPG, SAC
from src.Buffer import ReplayBuffer, PrioritizedReplayBuffer

ALGO_LEARNING_RATE = 3e-4
POLICY_LEARNING_RATE = 1e-4

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
        "lr": 0.1*POLICY_LEARNING_RATE,
        "shift_scale": 1.5,
        "min_log_scale": -10,
        "max_log_scale": 0.69,          # auto = size of action space
        "layer_units": (256, 256),
        "layer_kwargs": {
            "activation": tf_v1.nn.relu,
            "kernel_regularizer": None, # l2(1e-6),
            "kernel_constraint": max_norm(2.0),
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
        "layer_units": (50, 100, 50),
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
        "batch_size": 100,
        "layer_units": (50, 50, 50),
        "layer_kwargs": {
            "activation": tf_v1.nn.relu,
            "kernel_regularizer": l2(1e-6),
        }
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
        "lr": 0.99,
        "gamma": 0.996,
        "layer_units": (50, 50, 25),
        "tau": 0.01,
        "update_interval": 1,
        "layer_kwargs": {
            "activation": tf_v1.nn.relu,
            "kernel_regularizer": l2(1e-6),
        }
    }
}

_DDPG = {
    "function": DDPG,
    "kwargs": {
        "buffer_size": 500_000,
        "reward_scale": 1.0,
        "gamma": 0.99,
        "batch_size": 100,
        "layer_units": (100, 100, 50),
        "q_lr": POLICY_LEARNING_RATE,
        "tau": 0.008,
        "update_interval": 1,
        "random_type": "normal",
        "sigma": 0.3,           # Noise value = sigma * random.uniform(-1, 1) or random.normal(loc=0.0, scale=sigma)
        "layer_kwargs": {
            "activation": tf_v1.nn.relu,
            "kernel_regularizer": l2(1e-6),
        }
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
        "gamma": 0.9,
        "num_train": 2,
    }
}

_SAC = {
    "function": SAC,
    "kwargs": {
        "buffer_size": 500_000,
        "reward_scale": 1.0,
        "gamma": 0.99,
        "batch_size": 256,
        "num_init_exp_samples": 10000,
        "layer_units": (100, 100),
        "tau": 5e-3,            # param(targets) = tau*param(source) + (1-tau)*param(targets) 
        "update_interval": 1,   # Time steps
        "num_q_nets": 2,
        "q_lr": POLICY_LEARNING_RATE,
        "alpha_lr": POLICY_LEARNING_RATE,
        "auto_ent": True,
        "target_entropy": -2,    # auto = -0.5 * Action space size
        "init_log_alpha": 0.0,   # auto = 2 * Target entropy
        "layer_kwargs": {
            "activation": tf_v1.nn.relu,
            "kernel_regularizer": None # l2(1e-6),
            # "kernel_constraint": max_norm(2.0),
        }
    }
}

ALGORITHMS = {
    "dqn": _DQN,
    "ddqn": _DDQN,
    "reinforce": _Reinforce,
    "actor_critic": _ActorCritic,
    "sarsa": _Sarsa,
    "ddpg": _DDPG,
    "sac": _SAC,
}


def get_configuration(args):
    policy = copy.deepcopy(POLICIES[args.policy])
    algorithm = copy.deepcopy(ALGORITHMS[args.algorithm])
    algorithm["kwargs"].update(render=args.render)
    config_dict = {
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


def get_algorithm_from_variant(sess, env, policy, summary_dir, training, load_model, goal_trials, goal_reward, variant):
    param = variant["algorithm_param"]
    func, kwargs = get_func_and_kwargs_from_param(param)
    return func(sess=sess, env=env, policy=policy, summary_dir=summary_dir,
                training=training, goal_trials=goal_trials, goal_reward=goal_reward, load_model=load_model, **kwargs)
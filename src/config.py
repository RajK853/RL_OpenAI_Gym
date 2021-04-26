import copy

from src.Policy import GreedyEpsilonPolicy, GaussianPolicy, DiscretePolicy, ContinuousPolicy
from src.Algorithm import DQN, DDQN, Reinforce, A2C, Sarsa, DDPG, SAC
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
        "eps_kwargs": {
            "decay": 0.001,
            "clip_range": (0.001, 0.7),
        },
        "explore_ratio": 0.60,
        "explore_exploit_interval": 20,
    }
}

Gaussian = {
    "function": GaussianPolicy,
    "kwargs": {
        "lr_kwargs": {
            "type": "ConstantScheduler",
            "value": 0.0001,
        },
        "mu_range": (-2.0, 2.0),
        "log_std_range": (-20.0, -0.3),
    }
}

Discrete = {
    "function": DiscretePolicy,
    "kwargs": {
        "lr_kwargs": {
            "type": "ConstantScheduler",
            "value": 0.0001,
        },
    }
}

Continuous = {
    "function": ContinuousPolicy,
    "kwargs": {
        "lr_kwargs": {
            "type": "ConstantScheduler",
            "value": 0.0001,
        },
    }
}

POLICIES = {
    "greedy_epsilon": GreedyEpsilon,
    "gaussian": Gaussian,
    "discrete": Discrete,
    "continuous": Continuous,
}

# Algorithms
_Sarsa = {
    "function": Sarsa,
    "kwargs": {
        "gamma_kwargs": {
            "type": "ExpScheduler",
            "decay_rate": 0.005,
            "e_offset": 1.0,
            "e_scale": -1.0,
            "update_step": 50,
            "clip_range": (0.99, 0.997),
        },
        "lr_kwargs": {
            "type": "ExpScheduler",
            "decay_rate": 0.005,
            "update_step": 20,
            "clip_range": (0.0001, 0.001),
        },
    }
}

_DQN = {
    "function": DQN,
    "kwargs": {
        "gamma_kwargs": {
            "type": "ExpScheduler",
            "decay_rate": 0.005,
            "e_offset": 1.0,
            "e_scale": -1.0,
            "update_step": 50,
            "clip_range": (0.99, 0.997),
        },
        "lr_kwargs": {
            "type": "ExpScheduler",
            "decay_rate": 0.005,
            "update_step": 20,
            "clip_range": (0.0001, 0.001),
        },
        "batch_size_kwargs": {
            "type": "ConstantScheduler",
            "value":  100,
        },
        "num_init_exp_samples": 10000,
    }
}

_DDQN = {
    "function": DDQN,
    "kwargs": {
        "tau": 0.003,
        "update_interval": 1,
        "num_init_exp_samples": 1000,
        "max_init_exp_timestep": "auto",
        "batch_size_kwargs": {
            "type": "ConstantScheduler",
            "value":  100,
        },
        "gamma_kwargs": {
            "type": "ExpScheduler",
            "decay_rate": 0.005,
            "e_offset": 1.0,
            "e_scale": -1.0,
            "update_step": 50,
            "clip_range": (0.99, 0.997),
        },
        "lr_kwargs": {
            "type": "ExpScheduler",
            "decay_rate": 0.005,
            "update_step": 20,
            "clip_range": (0.0001, 0.001),
        },
    }
}

_DDPG = {
    "function": DDPG,
    "kwargs": {
        "num_init_exp_samples": 4000,
        "max_init_exp_timestep": 200,
        "buffer_size": 500_000,
        "reward_scale": 1.0,
        "tau": 0.003,
        "update_interval": 10,
        "sigma_kwargs": {
            "type": "ConstantScheduler",
            "value": 1.0,
        },
        "batch_size_kwargs": {
            "type": "ConstantScheduler",
            "value":  100,
        },
        "gamma_kwargs": {
            "type": "ConstantScheduler",
            "value": 0.99,
        },
        "lr_kwargs": {
            "type": "ConnstantScheduler",
            "value": 0.0001,
        },
    }
}

_Reinforce = {
    "function": Reinforce,
    "kwargs": {
        "alpha": 0.003,
        "gamma_kwargs": {
            "type": "ConstantScheduler",
            "value": 0.997,
        },
    }
}

_A2C = {
    "function": A2C,
    "kwargs": {
        "lr_kwargs": {
            "type": "ConstantScheduler",
            "value": 0.0001,
        },
        "gamma_kwargs": {
            "type": "ConstantScheduler",
            "value": 0.997,
        },
    }
}
_SAC = {
    "function": SAC,
    "kwargs": {
        "buffer_size": 500_000,
        "reward_scale": 1.0,
        "clip_norm": None,
        "num_init_exp_samples": 10000,
        "tau": 5e-3,              # param(targets) = tau*param(source) + (1-tau)*param(targets)
        "update_interval": 1,     # Time steps
        "num_q_nets": 2,
        "auto_ent": True,
        "target_entropy": "auto", # auto = -0.5 * Action space size
        "init_log_alpha": 0.0,    # TODO: Maybe specify init_alpha instead
        "gamma_kwargs": {
            "type": "ConstantScheduler",
            "value": 0.99,
        },        
        "batch_size_kwargs": {
            "type": "ConstantScheduler",
            "value":  256,
        },
        "q_lr_kwargs": {
            "type": "ConstantScheduler",
            "value": 0.0003,
        },
        "alpha_lr_kwargs": {
            "type": "ConstantScheduler",
            "value": 0.0003,
        },
    }
}


ALGORITHMS = {
    "dqn": _DQN,
    "ddqn": _DDQN,
    "reinforce": _Reinforce,
    "a2c": _A2C,
    "sarsa": _Sarsa,
    "ddpg": _DDPG,
    "sac": _SAC,
}



def _get_config(src_dict, key, deep=True):
    dict_value = src_dict.get(key, None)
    assert dict_value is not None, f"Invalid key '{key}' received!; Valid keys: {','.join(src_dict.keys())}"
    if deep:
        dict_value = copy.deepcopy(dict_value)
    return dict_value


def load_variant(src_dict, variant):
    config_dict = _get_config(src_dict, variant["name"], deep=True)
    config_dict["kwargs"].update(variant.get("kwargs", {}))
    return config_dict


def get_algo(variant):
    algo_config = load_variant(ALGORITHMS, variant)
    return algo_config["function"], copy.deepcopy(algo_config["kwargs"])


def get_policy(variant):
    policy_config = load_variant(POLICIES, variant)
    return policy_config["function"], copy.deepcopy(policy_config["kwargs"])
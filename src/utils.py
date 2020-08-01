import os
import logging
import argparse
import numpy as np
from importlib import reload
from datetime import datetime
from gym.spaces import Box, Discrete

VALID_ENVS = ("CartPole-v0", "LunarLander-v2", "MountainCar-v0")


def get_space_size(space):
    if isinstance(space, Box):
        return np.prod(space.shape, axis=0)
    elif isinstance(space, Discrete):
        return space.n
    else:
        raise NotImplementedError(f"Invalid space of type '{type(space)}'!")


def standardize_array(array):
    mean_value = np.mean(array)
    std_value = np.std(array)
    std_value = std_value if std_value > 0 else 1
    return (array-mean_value)/std_value


def normalize_array(array):
    max_value = max(array)
    min_value = min(array)
    return (array-min_value)/(max_value-min_value)


def dict2str(feed_dict, sep=", "):
    """
    Convert dictionary into a single string
    args:
        feed_dict (dict) : Dictionary to convert
        sep (str) : Separator to join pair of key:value
    returns:
        str : Converted dict as string
    """
    dict_strs = []
    for key, value in feed_dict.items():
        if isinstance(value, (np.float, np.float16, np.float32, np.float64)):
            dict_strs.append(f"{key}:{value:.3f}")
        else:
            dict_strs.append(f"{key}:{value}")
    return sep.join(dict_strs)


def get_logger(log_file):
    """
    Returns root logger object to log messages in a file and on console
    args:
        log_file (str) : Destination log file
    returns:
        logging.Logger : Logger object 
    """
    reload(logging)                             # Due to issue with creating root logger in Notebook
    logging.basicConfig(level=logging.DEBUG,
                        format="%(message)s",
                        datefmt="%m-%d %H:%M",
                        filename=log_file,
                        filemode="a")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(console)
    return logger


def eval_dict_values(config_dict):
    """
    Evaluates the values of the dictionary
    args:
        config_dict (dict) : Dictionary
    """
    return {key: eval(value) for key, value in config_dict.items()}


def boolean_string(s):
    """
    Checks if given string indicates boolean values True or False
    args:
        s (str) : String to check
    """
    s = s.lower()
    if s not in ('false', 'true'):
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def random_seed_gen(num):
    for _ in range(num):
        yield np.random.randint(0, 100)


def parse_args():
    """
    Parse arguments from command line
    returns:
         argparse.ArgumentParser : Parsed arguments
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--env_name", help="Open AI Gym environment name", type=str)
    arg_parser.add_argument("--seed_num", help="Random seed number", type=int, default=1)
    arg_parser.add_argument("--summ_dir", help="Summary directory", type=str, default=None)
    arg_parser.add_argument("--test_model_chkpt", help="Model checkpoint to test", type=str, default=None)
    arg_parser.add_argument("--epochs", help="Number of epochs", type=int, default=1000)
    arg_parser.add_argument("--record_interval", help="Record video at given epoch intervals", type=int, default=0)
    arg_parser.add_argument("--render", help="Render on the screen", type=boolean_string, default=True)
    arg_parser.add_argument("--display_interval", help="Display information on every given epoch", type=int, default=10)
    arg_parser.add_argument("--algorithm", help="Algorithm name", type=str, default="dqn")
    # TODO: 1) Remove policy, display_interval. 2) Rename seed_num. 3) Load goal information from a csv.
    #  4) Update csv with best hyper-parameter values
    arg_parser.add_argument("--policy", help="Policy name", type=str, default="greedy_epsilon")
    arg_parser.add_argument("--buffer", help="Replay buffer name", type=str, default="replay_buffer")
    arg_parser.add_argument("--goal_trials", help="Number of trials (epochs) to compute goal", type=int, default=100)
    arg_parser.add_argument("--goal_reward", help="Minimum goal reward value", type=float, default=200.0)

    _args = arg_parser.parse_args()
    env_name = _args.env_name
    if _args.summ_dir is None:
        date_time = datetime.now().strftime("%d.%m.%Y %H.%M")
        _args.summ_dir = os.path.join("summaries", f"{env_name} {date_time}")
    return _args
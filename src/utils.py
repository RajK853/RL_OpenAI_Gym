import os
import logging
import argparse
import numpy as np
from importlib import reload
from datetime import datetime
from gym.spaces import Box, Discrete


def polyak_average(src_value, target_value, tau=0.5):
    return target_value + tau*(src_value - target_value)


def get_space_size(space):
    if isinstance(space, Box):
        return np.prod(space.shape, axis=0)
    elif isinstance(space, Discrete):
        return space.n
    else:
        raise NotImplementedError(f"Invalid space of type '{type(space)}'!")


def standardize_array(array, default_std=1e-8):
    array = np.array(array)
    std_value = array.std()
    if std_value <= 0:
        std_value = default_std
    return (array-array.mean())/std_value


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
    reload(logging)                                        # Due to issue with creating root logger in Notebook
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


def get_goal_info(df, env_name):
    reward_cols = ["Trials", "rThresh"]
    env_cond = (df.loc[:, "Environment Id"] == env_name)
    assert any(env_cond), f"{env_name} not found!"
    goal_trials, goal_reward = df[env_cond].loc[:, reward_cols].to_numpy().squeeze()
    return int(goal_trials), float(goal_reward)


def parse_args():
    """
    Parse arguments from command line
    returns:
         argparse.ArgumentParser : Parsed arguments
    """
    # TODO: 1) Update csv with best hyper-parameter values, 2) Restoring model does not require algorithm
    #  or (load env_name, algo, policy from meta data)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--env_name", help="Open AI Gym environment name", type=str)
    arg_parser.add_argument("--num_exec", help="Number of executions", type=int, default=1)
    arg_parser.add_argument("--summ_dir", help="Summary directory", type=str, default=None)
    arg_parser.add_argument("--epochs", help="Number of epochs", type=int, default=1000)
    arg_parser.add_argument("--render", help="Render on the screen", type=boolean_string, default=False)
    arg_parser.add_argument("--record_interval", help="Video record interval (in epoch)", type=int, default=10)
    arg_parser.add_argument("--test_model_chkpt", help="Model checkpoint to evaluate", type=str, default=None)
    arg_parser.add_argument("--algorithm", help="Algorithm name", type=str, default="dqn")
    arg_parser.add_argument("--policy", help="Policy name", type=str, default="greedy_epsilon")

    _args = arg_parser.parse_args()
    env_name = _args.env_name
    if _args.summ_dir is None:
        date_time = datetime.now().strftime("%d.%m.%Y %H.%M")
        _args.summ_dir = os.path.join("summaries", f"{env_name} {date_time}")
    return _args

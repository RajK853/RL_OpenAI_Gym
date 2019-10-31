import os
import logging
import argparse
import numpy as np
import configparser
from importlib import reload
from itertools import product
from datetime import datetime

VALID_ENVS = ("CartPole-v0", "LunarLander-v2", "MountainCar-v0")


def dict2str(feed_dict, sep=", "):
    """
    Convert dictionary into a single string
    args:
        feed_dict (dict) : Dictionary to convert
        sep (str) : Seperator to join pair of key:value
    returns:
        str : Converted dict as string
    """
    dict_strs = []
    for key, value in feed_dict.items():
        if isinstance(value, (np.float, np.float16, np.float32, np.float64)):
            dict_strs.append("{}:{:.3f}".format(key, value))
        else:
            dict_strs.append("{}:{}".format(key, value))
    return sep.join(dict_strs)


def get_logger(log_file):
    """
    Returns root logger object to log messages in a file and on console
    args:
        log_file (str) : Destination log file
    returns:
        logging.Logger : Logger object 
    """
    reload(logging)  # Due to issue with creating root logger in Notebook
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
    return (s == 'true')


def parameter_generator(*args):
    """
    Generates different combination from given arguments
    args:
        Arguments of one of these typs: iterable objects (list, tuple, set, range), dictionary
    yields:
        list : List containing one value from each given arguments

    example:
    Input:
    for para in parameter_generator([1,2], (True, False), {"vowels" : "aeiou", "even":[2,4]}):
        print("para:", para)

    Output:

    para: [1, True, {'vowels': 'a', 'even': 2}]
    para: [1, True, {'vowels': 'a', 'even': 4}]
    para: [1, True, {'vowels': 'e', 'even': 2}]
    para: [1, True, {'vowels': 'e', 'even': 4}]
    .
    .
    para: [2, True, {'vowels': 'u', 'even': 4}]
    para: [2, True, {'vowels': 'o', 'even': 2}]
    para: [2, True, {'vowels': 'u', 'even': 4}]

    The order of returned parameter depends on the order they were given. 
    Dictionary arguments get back dictionary results with a single value in its respective key.
    """
    parameters = []
    parameter_keys = []
    # Argument preprocessing
    for arg in args:
        if isinstance(arg, dict):
            parameters.append(product(*arg.values()))
            parameter_keys.append(arg.keys())
        else:
            parameters.append(arg)
            parameter_keys.append(None)

    for parameter_values in product(*parameters):
        result = [val if keys is None else dict(zip(keys, val)) for keys, val in zip(parameter_keys, parameter_values)]
        yield result


def get_configuration(config_file):
    """
    Loads configuration file and returns its data as dict
    args:
        config_file (str) : INI Configuration file address
    returns:
        dict : Configuration data as dictionary
    """
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)
    # Load configurations from config file
    init_kwargs = eval_dict_values(config_parser["init_kwargs"])
    log_init_kwargs = eval_dict_values(config_parser["log_init_kwargs"])
    train_kwargs = eval_dict_values(config_parser["train_kwargs"])
    log_train_kwargs = eval_dict_values(config_parser["log_train_kwargs"])
    others = eval_dict_values(config_parser["others"])
    # Prepare configuration dictionary
    config_dict = {"kwargs": (init_kwargs, train_kwargs),
                   "log_kwargs": (log_init_kwargs, log_train_kwargs),
                   "others": others}
    return config_dict


def parse_args():
    """
    Parse arguments from command line
    returns:
         argparse.ArgumentParser : Parsed arguments
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--env_name", help="Open AI Gym environment name", type=str)
    arg_parser.add_argument("--log_file", help="Log file", type=str, default=None)
    arg_parser.add_argument("--summ_dir", help="Summary directory", type=str, default=None)
    arg_parser.add_argument("--config_file", help="INI Config file", type=str, default=None)
    arg_parser.add_argument("--plot_result", help="Plot result in matplotlib", type=boolean_string, default=False)
    arg_parser.add_argument("--test_model_chkpt", help="Model checkpoint to test", type=str, default=None)
    arg_parser.add_argument("--epochs", help="Number of epochs", type=int, default=1000)
    arg_parser.add_argument("--record_interval", help="Record video at given epoch intervals", type=int, default=0)
    arg_parser.add_argument("--render", help="Render on the screen", type=boolean_string, default=True)
    arg_parser.add_argument("--display_every", help="Display information on every given epoch", type=int, default=10)
    _args = arg_parser.parse_args()
    # Default summary directory, log file and config file
    date_time = datetime.now().strftime("%d.%m.%Y %H.%M")
    env_name = _args.env_name
    assert (env_name in VALID_ENVS), "Invalid environment received! Enter one of the followings:\n{}".format(VALID_ENVS)
    if _args.summ_dir is None:
        _args.summ_dir = os.path.join("summaries", "{} {}".format(env_name, date_time))
    if _args.log_file is None:
        _args.log_file = os.path.join(_args.summ_dir, "log", "Results {} {}.log".format(env_name, date_time))
    if _args.config_file is None:
        _args.config_file = os.path.join("config", "{}.ini".format(env_name))
    return _args

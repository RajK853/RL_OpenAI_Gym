import json
import yaml
import logging
import numpy as np
from importlib import reload
from gym.spaces import Box, Discrete

from . import Scheduler


def get_scheduler(config_dict):
    SchedulerClass = getattr(Scheduler, config_dict.pop("type"))
    return SchedulerClass(**config_dict)


def polyak_average(src_value, target_value, tau=0.5):
    return tau*src_value + (1-tau)*target_value


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


def json_dump(data_dict, file_name, **kwargs):
    with open(file_name, "w") as fp:
        json.dump(data_dict, fp=fp, **kwargs)


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
    if any(env_cond):
        assert any(env_cond), f"{env_name} not found!"
        goal_trials, goal_reward = df[env_cond].loc[:, reward_cols].to_numpy().squeeze()
        goal_reward = 0.0 if goal_reward == "None" else float(goal_reward)
    else:
        print(f"# Environment id {env_name} not found. Using default goal reward and trials value!")
        goal_reward = 0.0
        goal_trials = 1
    return int(goal_trials), goal_reward


def dump_yaml(data_dict, file_path, **kwargs):
    with open(file_path, "w") as fp:
        yaml.dump(data_dict, fp, **kwargs)


def load_yaml(file_path, safe_load=True):
    """
    Loads a YAML file from the given path
    :param file_path: (str) YAML file path
    :param safe_load: (bool) If True, uses yaml.safe_load() instead of yaml.load()
    :returns: (dict) Loaded YAML file as a dictionary
    """
    load_func = yaml.safe_load if safe_load else yaml.load
    with open(file_path, "r") as fp:
        return load_func(fp)


def exec_from_yaml(config_path, exec_func, title="Experiment", safe_load=True):
    """
    Executes the given function by loading parameters from a YAML file with given structure:
    - Experiment_1 Name:
        argument_1: value_1
        argument_2: value_2
        ...
    - Experiment_2 Name:
        argument_1: value_1
        argument_2: value_2
        ...
    NOTE: The argument names in the YAML file should match the argument names of the given execution function.
    :param config_path: (str) YAML file path
    :param exec_func: (callable) Function to execute with the loaded parameters
    :param title: (str) Label for each experiment
    :returns: (dict) Dictionary with results received from each experiment execution
    """
    result_dict = {}
    config_dict = load_yaml(config_path, safe_load=safe_load)
    i = 1
    for exp_name, exp_kwargs in config_dict.items():
        if exp_name.lower().startswith("default"):
            continue
        print(f"\n{i}. {title}: {exp_name}")
        # Execute the exec_func function by unpacking the experiment's keyword-arguments
        result = exec_func(**exp_kwargs)
        result_dict[exp_name] = result
        i += 1
    return result_dict

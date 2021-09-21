import os
import json
import yaml
import numpy as np
from gym.spaces import Box, Discrete

from . import Scheduler


def list_files(path, excludes=None, ftype=".py"):
    """
    Returns the list of files of given type present in the given directory
    :param path: (str) Directory path
    :param excludes: (list/tuple/iterator) File names to exclude
    :param ftype: (str) File extension
    :return: (list) List of files
    """
    if excludes is None:
        excludes = []
    filenames = [filename.rstrip(ftype) 
                 for filename in os.listdir(path) 
                 if filename.endswith(ftype) and filename not in excludes]
    return filenames


def get_scheduler(config_dict):
    """
    Returns a Scheduler object based on the config_dict
    :param config_dict: (dict) Dictionary with arguments for a given scheduler class
    :return: (Scheduler) Scheduler object
    """
    config_dict = config_dict.copy()
    SchedulerClass = getattr(Scheduler, config_dict.pop("type"))
    return SchedulerClass(**config_dict)


def get_space_size(space):
    if isinstance(space, Box):
        return np.prod(space.shape, axis=0)
    elif isinstance(space, Discrete):
        return space.n
    else:
        raise NotImplementedError(f"Invalid space of type '{type(space)}'!")


def polyak_average(src_value, target_value, tau=0.001):
    """
    Returns the Polyak average between src and target values
    :param src_value: (int/float) Source value
    :param target_value: (int/float) Target value
    :param tau: (float) Weight update coefficient. Soft update when tau < 1.0 and hard update when tau == 1.0
    :returns: (float) Polyak average value 
    """
    return tau*src_value + (1-tau)*target_value


def standardize_array(array):
    """
    Z-transformation of the given array
    :param array: (np.ndarray) Data array
    :returns: (np.ndarray) Standarized array
    """
    array = np.array(array)
    std_value = array.std()
    if std_value <= 0:
        std_value = 1e-8
    return (array-array.mean())/std_value


def normalize_array(array):
    """
    Min-Max normalization of the given array
    :param array: (np.ndarray) Data array
    :returns: (np.ndarray) Normalized array 
    """
    max_value = max(array)
    min_value = min(array)
    return (array-min_value)/(max_value-min_value)


def dict2str(feed_dict, sep=", "):
    """
    Converts dictionary into a single string
    :param feed_dict: (dict) Dictionary to convert
    :param sep: (str) Separator to join pair of key:value
    :returns: (str) Converted dict as string
    """
    dict_strs = []
    for key, value in feed_dict.items():
        if isinstance(value, (np.float, np.float16, np.float32, np.float64)):
            dict_strs.append(f"{key}:{value:.3f}")
        else:
            dict_strs.append(f"{key}:{value}")
    return sep.join(dict_strs)


def json_dump(data_dict, file_name, **kwargs):
    """
    Dumps given dictionary into a JSON file
    :param data_dict: (duct) Data dictionary
    :param file_path: (str) Dump JSON file path
    :param **kwargs: (dict) Additional kwargs for json.dump() function
    """
    with open(file_name, "w") as fp:
        json.dump(data_dict, fp=fp, **kwargs)


def random_seed_gen(num):
    """
    Generator to produce N random seeds
    :param num: (int) Number of random seeds
    :returns: (int) Random seed number
    """
    for _ in range(num):
        yield np.random.randint(0, 100)


def get_goal_info(df, env_name):
    """
    Returns goal info (goal trials and goal reward) from the given DataFrame
    :param df: (pandas.DataFrame) DataFrame with goal information of the environment
    :param env_name: (str) Name of the environment to retrive the goal information
    :returns: (int, float) Goal trials and goal reward
    """
    reward_cols = ["Trials", "rThresh"]
    env_cond = (df.loc[:, "Environment Id"] == env_name)
    if any(env_cond):
        assert any(env_cond), f"{env_name} not found!"
        goal_trials, goal_reward = df[env_cond].loc[:, reward_cols].to_numpy().squeeze()
        goal_reward = 0.0 if goal_reward == "None" else float(goal_reward)
    else:
        print(f"# Environment id {env_name} not found. Using default goal reward (0.0) and goal trials (1) values!")
        goal_reward = 0.0
        goal_trials = 1
    return int(goal_trials), goal_reward


def dump_yaml(data_dict, file_path, **kwargs):
    """
    Dumps given dictionary into a YAML file
    :param data_dict: (duct) Data dictionary
    :param file_path: (str) Dump YAML file path
    :param **kwargs: (dict) Additional kwargs for yaml.dump() function
    """
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


def load_files(dir_path, file_types=None):
    """
    Recursively searches and returns the file paths of given types located in the given directory path
    :param dir_path: (str) Root directory path to search
    :param file_types: (iter) Collection of file extension to search. Defaults to None
    :returns: (list) List of found file paths 
    """
    
    def has_type(file_name, types):
        if types is None:
            return True
        _, file_type = os.path.splitext(file_name)
        return file_type in types

    file_paths = []
    for root_dir, _, files in os.walk(dir_path):
        if files:
            matched_files = [os.path.join(root_dir, file_path) for file_path in files if has_type(file_path, file_types)]
            file_paths.extend(matched_files)
    return file_paths


def exec_from_yaml(config_path, exec_func, title="Experiment", safe_load=True, skip_prefix="ignore"):
    """
    Executes the given function by loading parameters from a YAML file with given structure:
    NOTE: The argument names in the YAML file should match the argument names of the given execution function.
    
    :Example:

    - Experiment_1 Name:
        argument_1: value_1
        argument_2: value_2
        ...
    - Experiment_2 Name:
        argument_1: value_1
        argument_2: value_2
        ...
    
    :param config_path: (str) YAML file path
    :param exec_func: (callable) Function to execute with the loaded parameters
    :param title: (str) Label for each experiment
    :param safe_load: (bool) When True, uses yaml.safe_load. Otherwise uses yaml.load
    :param skip_prefix: (str) Experiment names with given prefix will not be executed 
    :returns: (dict) Dictionary with results received from each experiment execution
    """
    i = 1
    result_dict = {}
    if os.path.isdir(config_path):
        yaml_paths = load_files(config_path, file_types=[".yaml"])
    elif config_path.endswith(".yaml"):
        yaml_paths = [config_path]
    else:
        raise NotImplementedError("Invalid config_path value. Must be either a path to a .yaml file or a path to directory with .yaml files!")
    for yaml_path in yaml_paths:
        config_dict = load_yaml(yaml_path, safe_load=safe_load)
        for exp_name, exp_kwargs in config_dict.items():
            if exp_name.lower().startswith(skip_prefix):
                print(f"# Skipped {exp_name}")
                continue
            print(f"\n{i}. {title}: {exp_name}")
            # Execute the exec_func function by unpacking the experiment's keyword-arguments
            result = exec_func(**exp_kwargs)
            result_dict[exp_name] = result
            i += 1
    return result_dict

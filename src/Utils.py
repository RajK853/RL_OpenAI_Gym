import logging
import numpy as np
from importlib import reload
from itertools import product

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
    reload(logging)          # Due to issue with creating root logger in Notebook
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
    return {key:eval(value) for key, value in config_dict.items()}

def boolean_string(s):
    """
    Checks if given string indicates boolean values True or False
    args:
        s (str) : String to check
    """
    s = s.lower()
    if s not in ('false', 'true'):
        raise ValueError('Not a valid boolean string')
    return (s=='true')

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
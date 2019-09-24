import os
import gym
import argparse
import warnings
import numpy as np
import configparser
from datetime import datetime
import matplotlib.pylab as plt
# Tensorflow
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
# Custom modules
from src import ReplayBuffer
from src import MountainCar
from src.Utils import get_logger, eval_dict_values, boolean_string

################################################ Local Functions ############################################################

def get_configuration(config_file):
    """
    Somehow in Jupyter Notebook config_parser keeps looking in ROOT_DIR\notebooks\config_dir 
    instead of ROOT_DIR\config_dir, even though the ROOT_DIR is in sys.path. 
    Therefore, explicit addressing is required in Jupyter Notebook.
    """
    # Parse configuration from config file
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
                   "log_kwargs" : (log_init_kwargs, log_train_kwargs), 
                   "others" : others}
    return config_dict

def run(env, seed, mem, logger, summ_dir, epochs, init_kwargs, train_kwargs, log_init_kwargs, log_train_kwargs, 
        plot_result, *, model_i=1, sess_config=None, test_model_chkpt=None):
    """
    Runs the simulation with given parameters
    args:
        env (gym.Env/gym.wrappers.Monitor) : Normal or wrapped gym environment
        seed (int) : Seed value
        mem (src.ReplayBuffer) : ReplayBuffer object
        logger (logging.Logger) : Logger object
        summ_dir (src) : Summary directory
        init_kwargs (dict) : Dictionary with keyworded arguments to initialize agent
        train_kwargs (dict) : Dictionary with keyworded arguments to train agent
        log_init_kwargs (dict) : Other init_kwargs that will be logged
        log_train_kwargs (dict) : Other train_kwargs that will be logged
        plot_result (bool) : If True, plots summary results of all episodes using matplotlib
        model_i (int) : Model index
        sess_config (tf_v1.ConfigProto) : Tensorflow configuration protocol for Session object
        test_model_chkpt (str) : Model checkpoint to restore and test variables of the model for agent 
    """
    env.seed(int(seed))
    model_name = "Model {}".format(model_i)
    # Create summary directory
    _summ_dir = os.path.join(summ_dir, model_name)
    training = test_model_chkpt is None
    with tf_v1.Session(config=sess_config) as sess:
        agent = MountainCar(sess, env, mem, summ_dir=_summ_dir, **init_kwargs, **log_init_kwargs)
        sess.run(tf_v1.global_variables_initializer())
        if training:
            print("\n# Training: {}".format(model_name))
        else:
            print("\n# Testing: {}".format(test_model_chkpt))
            # Restore model variables
            agent.restore_model(test_model_chkpt)
        # Run (test/train) the agent
        results = agent.run(**train_kwargs, **log_train_kwargs,  epochs=epochs, training=training)
        if training:
            # Save model as checkpoint
            agent.save_model(os.path.join(_summ_dir, "model.chkpt"))
        *results, goal_summary = results
        if plot_result:                                         # Plot results in matplotlib
            warnings.warn("""The program will be paused while matplotlib window is opened. 
                Therefore, unless required, set plot_result value to False and use tensorboard instead.""")
            rows = len(results)
            for i, (p, plt_name) in enumerate(zip(results, ("Losses", "Rewards", "Max pos", "Epsilons")), start=1):
                plt.subplot(rows, 1, i)
                plt.plot(p)
                plt.xlabel('Epochs')
                plt.ylabel(plt_name)
            plt.show()
    # Clear replay buffer and tensorflow graph
    mem.clear()
    tf_v1.reset_default_graph()
    # Log parameters and achieved goals information
    parameter_dict = {"seed":seed, **log_init_kwargs, **log_train_kwargs, "training":training}
    agent.log(logger, model_name, parameter_dict, goal_summary)

def main():
    # Default summary directoy, log and config file
    date_time = datetime.now().strftime("%d.%m.%Y %H.%M")
    default_summ_dir = os.path.join(SUMM_DIR, "{} {}".format(ENV_NAME, date_time))
    default_log_file = os.path.join(default_summ_dir, LOG_DIR, "Results {} {}.log".format(ENV_NAME, date_time))
    default_config_file = os.path.join(CONFIG_DIR, "LunarLander-v2.ini")
    # Parse arguments from command line
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--log_file", help="Log file", type=str, default=default_log_file)
    arg_parser.add_argument("--summ_dir", help="Summary directory", type=str, default=default_summ_dir)
    # TODO: Better if config file address is hard coded?
    arg_parser.add_argument("--config_file", help="ini Config file", type=str, default=default_config_file)
    arg_parser.add_argument("--plot_result", help="Plot result in matplotlib", type=boolean_string, default=False)
    arg_parser.add_argument("--test_model_chkpt", help="Model checkpoint to test", type=str, default=None)
    arg_parser.add_argument("--epochs", help="Number of epochs", type=int, default=1000)
    arg_parser.add_argument("--record_interval", help="Record video at given epoch intervals", type=int, default=0)
    args = arg_parser.parse_args()
    # Load and unpack configurations
    config_dict = get_configuration(args.config_file)
    init_kwargs, train_kwargs = config_dict["kwargs"]
    log_init_kwargs, log_train_kwargs = config_dict["log_kwargs"]
    mem_size = config_dict["others"]["mem_size"]
    # Setup logger directory
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    # Create environment, replay buffer and buffer
    env = gym.make(ENV_NAME)
    mem = ReplayBuffer(mem_size)
    logger = get_logger(args.log_file)
    # Wrap environment to record videos
    if args.record_interval > 0:
        env = gym.wrappers.Monitor(env, os.path.join(args.summ_dir, "videos"), force=True,
                                   video_callable=lambda epoch: not epoch%args.record_interval)
    if args.test_model_chkpt is not None:
        # Override goal_trials and display_every parameters
        train_kwargs["goal_trials"] = 1
        train_kwargs["display_every"] = args.record_interval if (args.record_interval > 0) else args.epochs/10
    # Run model
    for model_i, seed in enumerate(SEEDS, start=1):
        run(env, seed, mem, logger, args.summ_dir, args.epochs, init_kwargs, train_kwargs, log_init_kwargs, log_train_kwargs, 
            args.plot_result, model_i=model_i, sess_config=TF_CONFIG, test_model_chkpt=args.test_model_chkpt)

################################################ End of Local Functions #######################################################

# Global variables
LOG_DIR = "log"
SUMM_DIR = "summaries"
CONFIG_DIR = "config"
ENV_NAME = "LunarLander-v2"
SEEDS = np.random.randint(100, 1000, size=1, dtype=np.uint16)
TF_CONFIG = tf_v1.ConfigProto(gpu_options=tf_v1.GPUOptions(per_process_gpu_memory_fraction=0.5), 
                              allow_soft_placement=True)

if __name__ == "__main__":
    main()
import os
import gym
import warnings
import numpy as np
import matplotlib.pylab as plt
import tensorflow.compat.v1 as tf_v1
from functools import partial
# Custom modules
from src import ReplayBuffer, Agent
from src.Utils import get_logger, parse_args, get_configuration


def run(env, seed, mem, logger, summ_dir, epochs, render, display_every, init_kwargs, train_kwargs, log_init_kwargs, log_train_kwargs,
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
    model_summ_dir = os.path.join(summ_dir, model_name)
    training = test_model_chkpt is None
    with tf_v1.Session(config=sess_config) as sess:
        agent = Agent(sess, env, mem, summ_dir=model_summ_dir, render=render, **init_kwargs, **log_init_kwargs)
        sess.run(tf_v1.global_variables_initializer())
        if training:
            print("\n# Training: {}".format(model_name))
        else:
            print("\n# Testing: {}".format(test_model_chkpt))
            agent.restore_model(test_model_chkpt)                       # Restore model variables
        # Run (test/train) the agent
        results = agent.run(epochs=epochs, training=training, display_every=display_every, **train_kwargs, **log_train_kwargs)
        if training:
            # Save model as checkpoint
            agent.save_model(os.path.join(model_summ_dir, "model.chkpt"))
        *results, goal_summary = results
        if plot_result:                                                 # Plot results in matplotlib
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
    parameter_dict = {"seed": seed, **log_init_kwargs, **log_train_kwargs, "training": training}
    agent.log(logger, model_name, parameter_dict, goal_summary)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
SEEDS = np.random.randint(100, 1000, size=1, dtype=np.uint16)
TF_CONFIG = tf_v1.ConfigProto(gpu_options=tf_v1.GPUOptions(per_process_gpu_memory_fraction=0.5),
                              allow_soft_placement=True)
if __name__ == "__main__":
    args = parse_args()
    # Load and unpack configurations
    config_dict = get_configuration(args.config_file)
    init_kwargs, train_kwargs = config_dict["kwargs"]
    log_init_kwargs, log_train_kwargs = config_dict["log_kwargs"]
    mem_size = config_dict["others"]["mem_size"]
    # Setup logger directory
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    # Create environment, replay buffer and buffer
    env = gym.make(args.env_name)
    mem = ReplayBuffer(mem_size)
    logger = get_logger(args.log_file)
    # Wrap environment to record videos
    if args.record_interval > 0:
        env = gym.wrappers.Monitor(env, os.path.join(args.summ_dir, "videos"), force=True,
                                   video_callable=lambda epoch: not epoch % args.record_interval)
    sim_func = partial(run, sess_config=TF_CONFIG, env=env, mem=mem, logger=logger, summ_dir=args.summ_dir,
                       epochs=args.epochs, render=args.render, display_every=args.display_every,
                       plot_result=args.plot_result, init_kwargs=init_kwargs, train_kwargs=train_kwargs,
                       log_init_kwargs=log_init_kwargs, log_train_kwargs=log_train_kwargs,
                       test_model_chkpt=args.test_model_chkpt)
    # Run model
    for model_i, seed in enumerate(SEEDS, start=1):
        sim_func(seed=seed, model_i=model_i)

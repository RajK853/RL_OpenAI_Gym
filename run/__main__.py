import os
import gym
from pandas import read_csv
from datetime import datetime
import tensorflow.compat.v1 as tf_v1
from tensorflow.python.util import deprecation
# Custom modules
from src.utils import parse_args, random_seed_gen, get_reward_info
from src.config import get_configuration, get_algorithm_from_variant, get_buffer_from_variant, get_policy_from_variant

deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
TF_CONFIG = tf_v1.ConfigProto(gpu_options=tf_v1.GPUOptions(per_process_gpu_memory_fraction=0.5),
                              allow_soft_placement=True)


def run(env, seed, model_i, *, summary_dir, cmd_args, goal_info, config, sess_config=None):
    """
    Runs the simulation with given parameters
    args:
        env (gym.Env/gym.wrappers.Monitor) : Normal or wrapped gym environment
        seed (int) : Seed value
        summary_dir (src) : Summary directory
        model_i (int) : Model index
        sess_config (tf_v1.ConfigProto) : Tensorflow configuration protocol for Session object
    """
    info_text = ("  Training information are available via Tensorboard with the given command:\n"
                 "  tensorboard --logdir summaries --host localhost")
    env.seed(seed)
    model_name = f"Model {model_i}"
    training = cmd_args.test_model_chkpt is None
    model_summary_dir = os.path.join(summary_dir, model_name) if training else None
    with tf_v1.Session(config=sess_config) as sess:
        buffer = get_buffer_from_variant(config)
        policy = get_policy_from_variant(env, config)
        algo = get_algorithm_from_variant(sess, env, policy, buffer, model_summary_dir, training, *goal_info, config)
        sess.run(tf_v1.global_variables_initializer())
        if training:
            print("\n# Training: {}".format(model_name))
            if training:
                print(info_text)
        else:
            print("\n# Testing: {}".format(cmd_args.test_model_chkpt))
            algo.restore_model(cmd_args.test_model_chkpt)          # Restore model variables
        algo.run(total_epochs=cmd_args.epochs)                     # Run the algorithm for given epochs
        if training:                                               # Save trained model as checkpoint
            algo.save_model(os.path.join(model_summary_dir, "model.chkpt"))
    buffer.clear()
    tf_v1.reset_default_graph()
    # TODO: Which informations to log?
    # Log parameters and achieved goals information
    # parameter_dict = {"seed": seed, "training": training}
    # algo.log(logger, model_name, parameter_dict, goal_summary)


if __name__ == "__main__":
    env_info_file = f"./assets/env_info.csv"
    date_time = datetime.now().strftime("%d.%m.%Y %H.%M")
    cmd_args = parse_args()
    summary_dir = os.path.join("summaries", f"{cmd_args.env_name}-{cmd_args.algorithm}-{cmd_args.policy}-{date_time}")
    env = gym.make(cmd_args.env_name)
    # Wrap environment to record videos
    if cmd_args.record_interval > 0:
        env = gym.wrappers.Monitor(env, os.path.join(summary_dir, "videos"), force=True,
                                   video_callable=lambda epoch: (not epoch % cmd_args.record_interval) or
                                                                (epoch == cmd_args.epochs-1))
    config_dict = get_configuration(cmd_args)
    df = read_csv(env_info_file)
    goal_info = get_reward_info(df, env_name=cmd_args.env_name)
    del df
    # Run model
    for model_i, seed in enumerate(random_seed_gen(cmd_args.seed_num), start=1):
        run(seed=seed, model_i=model_i, cmd_args=cmd_args, sess_config=TF_CONFIG, env=env,
            summary_dir=summary_dir, goal_info=goal_info, config=config_dict)

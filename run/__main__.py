import os
import gym
from copy import deepcopy
from pandas import read_csv
from datetime import datetime
import tensorflow.compat.v1 as tf_v1
from tensorflow.python.util import deprecation
# Custom modules
from src.utils import parse_args, random_seed_gen, get_goal_info, json_dump
from src.config import get_configuration, get_algorithm_from_variant, get_policy_from_variant

deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
TF_CONFIG = tf_v1.ConfigProto(gpu_options=tf_v1.GPUOptions(per_process_gpu_memory_fraction=0.5),
                              allow_soft_placement=True)


def is_pybullet_env(env_name):
    return "BulletEnv" in env_name


def process_config(data_dict):
    for key, value in data_dict.items():
        if type(value) is dict:
            value = process_config(value)
        elif isinstance(value, type) or callable(value):
            value = value.__name__ if hasattr(value, "__name__") else value.__class__.__name__
        data_dict[key] = value
    return data_dict


def save_configuration(dump_file, dump_dict):
    dump_dict = process_config(deepcopy(dump_dict))
    json_dump(dump_dict, file_name=dump_file, indent=4)
    print(f"# Configurations saved at '{dump_file}'")


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
    env.seed(seed)
    model_name = f"Model {model_i}"
    training = cmd_args.training
    model_summary_dir = os.path.join(summary_dir, model_name) if training else None
    tf_v1.reset_default_graph()
    with tf_v1.Session(config=sess_config) as sess:
        policy = get_policy_from_variant(env, config)
        algo = get_algorithm_from_variant(sess, env, policy, model_summary_dir, training, cmd_args.load_model, *goal_info, config)
        if training:
            print("\n# Training: {}".format(model_name))
            print("  Training information are available via Tensorboard with the given command:\n"
                 "  tensorboard --logdir summaries --host localhost")
        else:
            print("\n# Testing: {}".format(cmd_args.load_model))
        algo.run(total_epochs=cmd_args.epochs)                     # Run the algorithm for given epochs
        if training:                                               # Save trained model as checkpoint
            algo.save_model(os.path.join(model_summary_dir, "model.chkpt"))


if __name__ == "__main__":
    try:
        env_info_file = f"./assets/env_info.csv"
        date_time = datetime.now().strftime("%d.%m.%Y %H.%M")
        cmd_args = parse_args()
        if is_pybullet_env(cmd_args.env_name):
            import pybullet_envs
            goal_info = (100, 0)
            print(f"# PyBullet environment '{cmd_args.env_name}' detected! "
                  f"Using the default goal trial {goal_info[0]} and threshold reward {goal_info[1]}")
        else:
            df = read_csv(env_info_file)
            goal_info = get_goal_info(df, env_name=cmd_args.env_name)
            del df
        env = gym.make(cmd_args.env_name)
        summary_dir = os.path.join("summaries", f"{cmd_args.env_name}-{cmd_args.algorithm}-{cmd_args.policy}-{date_time}")

        def video_callable(epoch):
            if cmd_args.record_interval > 0:
                return ((epoch + 1) % cmd_args.record_interval) == 0
            return False
        # Wrap environment to record videos
        env = gym.wrappers.Monitor(env, os.path.join(summary_dir, "videos"), force=True, video_callable=video_callable)
        config_dict = get_configuration(cmd_args)
        save_configuration(os.path.join(summary_dir, "config.json"), config_dict)
        # Run model
        for model_i, seed in enumerate(random_seed_gen(cmd_args.num_exec), start=1):
            run(seed=seed, model_i=model_i, cmd_args=cmd_args, sess_config=TF_CONFIG, env=env,
                summary_dir=summary_dir, goal_info=goal_info, config=config_dict)
    except Exception as ex:
        raise ex
    finally:
        tf_v1.reset_default_graph()

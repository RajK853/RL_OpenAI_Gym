import os
import sys
import gym
import random
from datetime import datetime
import tensorflow.compat.v1 as tf_v1
from tensorflow.python.util import deprecation

from src.config import get_algo, get_policy
from src.utils import load_yaml, exec_from_yaml, get_goal_info

deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
TF_CONFIG = tf_v1.ConfigProto(gpu_options=tf_v1.GPUOptions(per_process_gpu_memory_fraction=0.5),
                              allow_soft_placement=True)


def load_goal_info(env_name):
    if "BulletEnv" in env_name:
        import pybullet_envs
        goal_trials, goal_reward = (100, 0)
        print(f"# PyBullet environment '{env_name}' detected! Using the default goal trial {goal_trials} "
              f"and threshold reward {goal_reward}")
        print("# NOTE: PyBullet environments cannot be rendered! Please look at the recorded videos to observe it!")
    else:
        from pandas import read_csv
        env_info_file = r"assets/env_info.csv"
        df = read_csv(env_info_file)
        goal_trials, goal_reward = get_goal_info(df, env_name=env_name)
    return goal_trials, goal_reward


def get_env(env_name, seed, record_interval=0, dump_dir=None):
    def record_video(epoch):
        return ((epoch + 1) % record_interval) == 0

    env = gym.make(env_name)
    if record_interval > 0:
        assert dump_dir is not None, "Directory path to store the recorded videos not specified!"
        env = gym.wrappers.Monitor(env, dump_dir, force=True, video_callable=record_video)
    env.seed(seed)
    return env


def main(env_name, algo, policy, epochs, training=True, record_interval=10, seed=None, load_model=None, render=False):
    seed = random.randint(0, 100) if seed is None else seed
    date_time = datetime.now().strftime("%d.%m.%Y_%H.%M")
    goal_trials, goal_reward = load_goal_info(env_name)
    summary_dir = os.path.join("summaries", f"{env_name}-{algo['name']}-{policy['name']}-{date_time}")
    os.makedirs(summary_dir, exist_ok=True)
    env = get_env(env_name, seed, record_interval, dump_dir=os.path.join(summary_dir, "videos"))
    tf_v1.reset_default_graph()
    with tf_v1.Session(config=TF_CONFIG) as sess:
        policy["kwargs"].update({"env": env})
        _policy = get_policy(policy)
        model_summary_dir = os.path.join(summary_dir, "model") if training else None
        algo["kwargs"].update({"policy": _policy,
                               "sess": sess,
                               "env": env,
                               "summary_dir": model_summary_dir,
                               "training": training,
                               "load_model": load_model,
                               "goal_trials": goal_trials,
                               "goal_reward": goal_reward,
                               "render": render})
        _algo = get_algo(algo)
        if training:
            print("\n# Training:")
            print("  Training information are available via Tensorboard with the given command:\n"
                  "  tensorboard --logdir summaries --host localhost")
        else:
            print(f"\n# Testing: {load_model}")
        _algo.run(total_epochs=epochs)                     # Run the algorithm for given epochs
        if training:                                               # Save trained model as checkpoint
            model_chkpt_path = os.path.join(model_summary_dir, "model.chkpt")
            _algo.save_model(model_chkpt_path)
    env.close()


if __name__ == "__main__":
    yaml_config_file = sys.argv[1]
    exec_from_yaml(yaml_config_file, exec_func=main)

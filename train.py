import os
import sys
import gym
import random
from datetime import datetime
import tensorflow.compat.v1 as tf_v1
from tensorflow.python.util import deprecation
from pandas import read_csv

from src.config import get_algo, get_policy
from src.utils import exec_from_yaml, get_goal_info, dump_yaml

deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
TF_CONFIG = tf_v1.ConfigProto(gpu_options=tf_v1.GPUOptions(), allow_soft_placement=True)

ENV_INFO_FILE = r"assets/env_info.csv"

def load_goal_info(env_name):
    df = read_csv(ENV_INFO_FILE)
    goal_trials, goal_reward = get_goal_info(df, env_name=env_name)
    return goal_trials, goal_reward


def get_env(env_name, *, dump_dir, seed=None, record_interval=0, include=None, pre_render=False):
    def record_video(epoch):
        return ((epoch + 1) % record_interval) == 0
    # Import any required gym environment module
    if include is not None:
        import importlib
        print(f"# Importing: {include}")
        for module in include:
            importlib.import_module(module)
    # Load the environment and apply required wrappers
    env = gym.make(env_name)
    if seed is not None:
        env.seed(seed)
    if pre_render:
        env.render()
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FilterObservation(env, filter_keys=["observation", "desired_goal"])  # TODO: Take filter_keys as arguments?
        env = gym.wrappers.FlattenObservation(env)
    video_callable = record_video if record_interval > 0 else lambda x: False
    env = gym.wrappers.Monitor(env, dump_dir, force=True, video_callable=video_callable)
    return env


def main(env_name, algo, policy, epochs, training=True, record_interval=10, seed=None, load_model=None, render=False,
         summary_dir="summaries", include=None):
    seed = random.randint(0, 100) if seed is None else seed
    date_time = datetime.now().strftime("%d.%m.%Y_%H.%M")
    goal_trials, goal_reward = load_goal_info(env_name)
    exp_summary_dir = os.path.join(summary_dir, f"{env_name}-{algo['name']}-{policy['name']}-{date_time}")
    os.makedirs(exp_summary_dir, exist_ok=True)
    video_dump_dir = os.path.join(exp_summary_dir, "videos")
    env = get_env(env_name, seed=seed, record_interval=record_interval, dump_dir=video_dump_dir, include=include)
    model_summary_dir = os.path.join(exp_summary_dir, "model") if training else None
    tf_v1.reset_default_graph()
    with tf_v1.Session(config=TF_CONFIG) as sess:
        try:
            algo_func, algo_kwargs = get_algo(algo)
            policy_func, policy_kwargs = get_policy(policy)

            if training:
                print("\n# Training:")
                print("  Training information are available via Tensorboard with the given command:")
                print(f"  tensorboard --host localhost --logdir {summary_dir}")
                config_dict = {"epochs": epochs, 
                               "seed": seed, 
                               "load_model": load_model, 
                               "algo": {"name": algo["name"], 
                                        "kwargs": algo_kwargs},
                               "policy": {"name": policy["name"], "kwargs": policy_kwargs}}
                dump_yaml(config_dict, file_path=os.path.join(exp_summary_dir, "config.yaml"))
            else:
                print(f"\n# Testing: {load_model}")

            policy_kwargs.update({"env": env})
            _policy = policy_func(**policy_kwargs)
            
            algo_kwargs.update({"policy": _policy,
                                "sess": sess,
                                "env": env,
                                "summary_dir": model_summary_dir,
                                "training": training,
                                "load_model": load_model,
                                "goal_trials": goal_trials,
                                "goal_reward": goal_reward,
                                "render": render,
                                "seed": seed})
            _algo = algo_func(**algo_kwargs)
            _algo.run(total_epochs=epochs)                     # Run the algorithm for given epochs
        except KeyboardInterrupt:
            print("\n# Interrupted by the user!")
            _algo.hook_after_train()
        finally:
            env.close()


if __name__ == "__main__":
    yaml_config_file = sys.argv[1]
    exec_from_yaml(yaml_config_file, exec_func=main, safe_load=True)

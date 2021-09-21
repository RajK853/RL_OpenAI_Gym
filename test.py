import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tqdm
import argparse
import numpy as np
from datetime import datetime
from gym.spaces import Box, Discrete
import tensorflow.compat.v1 as tf_v1

from train import get_env


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, help="Environment name", required=True)
    parser.add_argument("--load_model", type=str, help=".tf model path", required=True)
    parser.add_argument("--dump_path", type=str, help="Path to dump recorded videos", default=None)
    parser.add_argument("--epochs", type=int, help="Number of test epochs", default=10)
    parser.add_argument("--include", help="Additional modules to import", nargs="*")
    args = parser.parse_args()
    arg_dict = args.__dict__
    return arg_dict
    

def discrete_policy_wrapper(policy):
    def wrapper(state, **kwargs):
        action = policy.predict(np.expand_dims(state, axis=0))[0]
        return np.argmax(action)
    return wrapper


def continuous_policy_wrapper(policy):
    def wrapper(state, **kwargs):
        action = policy.predict(np.expand_dims(state, axis=0))[0]
        return action
    return wrapper


def wrap_policy(env, policy):
    if isinstance(env.action_space, Discrete):
        policy = discrete_policy_wrapper(policy)
    elif isinstance(env.action_space, Box):
        policy = continuous_policy_wrapper(policy)
    else:
        raise NotImplementedError
    return policy


def setup(env_name, load_model, dump_path, include):
    env = get_env(env_name, record_interval=1, dump_dir=dump_path, include=include)
    policy = tf_v1.keras.models.load_model(load_model)
    policy = wrap_policy(env, policy)
    return env, policy


def rollout(env, policy, epochs=1):
    for epoch in tqdm.trange(1, epochs+1, desc="Testing"):
        done = False
        state = env.reset()
        while not done:
            action = policy(state)
            next_state, reward, done, info = env.step(action)
            state = next_state


def main(env_name, load_model, dump_path=None, epochs=5, include=None):
    if dump_path is None:
        date_time = datetime.now().strftime("%d.%m.%Y_%H.%M")
        dump_path = os.path.join("test_videos", f"{env_name}-{date_time}")
    env, policy = setup(env_name, load_model, dump_path, include)
    rollout(env, policy, epochs=epochs)


if __name__ == "__main__":
    arg_dict = get_args()
    main(**arg_dict)

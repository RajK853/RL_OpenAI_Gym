import os
import gym
from datetime import datetime
import tensorflow.compat.v1 as tf_v1
# Custom modules
from src.Utils import parse_args, random_seed_gen
from src.Config import get_configuration, get_algorithm_from_variant, get_buffer_from_variant, get_policy_from_variant


def run(env, seed, model_i, *, summary_dir, cmd_args, config, sess_config=None):
    """
    Runs the simulation with given parameters
    args:
        env (gym.Env/gym.wrappers.Monitor) : Normal or wrapped gym environment
        seed (int) : Seed value
        summary_dir (src) : Summary directory
        model_i (int) : Model index
        sess_config (tf_v1.ConfigProto) : Tensorflow configuration protocol for Session object
    """
    env.seed(int(seed))
    model_name = f"Model {model_i}"
    # Create summary directory
    model_summary_dir = os.path.join(summary_dir, model_name)
    training = cmd_args.test_model_chkpt is None
    with tf_v1.Session(config=sess_config) as sess:
        # TODO: Policy and algorithm hardcoded
        buffer = get_buffer_from_variant(config)
        policy = get_policy_from_variant(env, config)
        # TODO: Parameters hardcoded
        algo = get_algorithm_from_variant(sess, env, policy, buffer, model_summary_dir, config)
        sess.run(tf_v1.global_variables_initializer())
        if training:
            print("\n# Training: {}".format(model_name))
        else:
            print("\n# Testing: {}".format(cmd_args.test_model_chkpt))
            algo.restore_model(cmd_args.test_model_chkpt)                       # Restore model variables
        # Run the algorithm for given epochs
        algo.run(epochs=cmd_args.epochs)
        if training:
            # Save model as checkpoint
            algo.save_model(os.path.join(model_summary_dir, "model.chkpt"))
    # Clear replay buffer and tensorflow graph
    buffer.clear()
    tf_v1.reset_default_graph()
    # TODO: Which informations to log?
    # Log parameters and achieved goals information
    # parameter_dict = {"seed": seed, "training": training}
    # algo.log(logger, model_name, parameter_dict, goal_summary)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
TF_CONFIG = tf_v1.ConfigProto(gpu_options=tf_v1.GPUOptions(per_process_gpu_memory_fraction=0.5),
                              allow_soft_placement=True)
if __name__ == "__main__":
    date_time = datetime.now().strftime("%d.%m.%Y %H.%M")
    cmd_args = parse_args()
    summary_dir = os.path.join("summaries", f"{cmd_args.env_name}-{cmd_args.algorithm}-{cmd_args.policy}-{date_time}")
    env = gym.make(cmd_args.env_name)
    # Wrap environment to record videos
    if cmd_args.record_interval > 0:
        env = gym.wrappers.Monitor(env, os.path.join(summary_dir, "videos"), force=True,
                                   video_callable=lambda epoch: not epoch % cmd_args.record_interval)
    config_dict = get_configuration(cmd_args)
    # Run model
    for model_i, seed in enumerate(random_seed_gen(cmd_args.seed_num), start=1):
        run(seed=seed, model_i=model_i, cmd_args=cmd_args, sess_config=TF_CONFIG, env=env,
            summary_dir=summary_dir, config=config_dict)

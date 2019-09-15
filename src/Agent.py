import os
import numpy as np
from time import time
from datetime import datetime
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
# Local modules
from src import Estimator
from src.Utils import dict2str

class MountainCar:
    """
    MountainCar-v0 agent
    """
    steps = 0
    def __init__(self, sess, env, memory, batch_size=100, eps_limits=None, eps_decay=0.25, df=0.97, lr=0.9, tau=0.01,\
                 ddqn=False, render=False, summ_dir=None):
        """
        Constructor function
        args:
            sess (tf_v1.Session) : Tensorflow session object
            env (gym.Env) : OpenAI Gym environemnt object
            memory (src.ReplayBuffer) : ReplayBuffer object
            batch_size (int) : Maximum batch size for training the model
            eps_limits (float, float) : Maximum and minimum epsilon values
            eps_decay (float) : Decay ratio for epsilon after each epoch
            df (float) : Discount factor for Q-learning formula
            lr (float) : Learning rate for Q-learning formula
            tau (float) : Tau value for weight transfer in DDQN
            ddqn (bool) : DQN or Double DQN
            render (bool) : Render parameter for the simulation
            summ_dir (str) : Directory for tensorboard summary
        """
        self.steps = 0
        self.sess = sess
        self.replay_buffer = memory
        # Training parameters
        self.eps_limits = (1, 0.005) if eps_limits is None else eps_limits
        self.eps_decay = eps_decay
        self.eps = eps_limits[0]
        self.df = df
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        # Environment settings
        self.env = env
        self.render = render
        self.num_actions = self.env.action_space.n
        self.state_shape = self.env.observation_space.shape
        # Estimator settings
        self.ddqn = ddqn
        self.local_estimator = Estimator(self.state_shape, self.num_actions, scope="Local_network")
        self.q_vars = tf_v1.get_collection(tf_v1.GraphKeys.TRAINABLE_VARIABLES, scope="Local_network")
        if ddqn:
            self.target_estimator = Estimator(self.state_shape, self.num_actions, scope="Target_network")
            self.target_vars = tf_v1.get_collection(tf_v1.GraphKeys.TRAINABLE_VARIABLES, scope="Target_network")
            with tf_v1.name_scope("Weights_updater"):
                self.weights_updater = tf.group([tf_v1.assign(t_var,t_var+self.tau*(q_var-t_var)) for t_var, q_var in zip(self.target_vars, self.q_vars)])
        else:
            self.target_estimator = self.local_estimator
            self.target_vars = None
            self.weights_updater = None
        # Setup summaries
        self.summ_dir = summ_dir
        self.summ_writer = None if summ_dir is None else tf_v1.summary.FileWriter(summ_dir, sess.graph)
        self.summaries = self.setup_summaries()
        # Model saver
        self.saver = tf_v1.train.Saver(max_to_keep=10)

    def setup_summaries(self):
        """
        Setup the summary placeholders and their scalars Tensors
        args:
            None
        returns:
            Tensor : Tensor with all summary scalars merged into
        """
        with tf.name_scope("Performance_Summary"):
            self.epoch_reward_ph = tf_v1.placeholder(tf.float32, name="epoch_reward_ph")
            self.epoch_avg_loss_ph = tf_v1.placeholder(tf.float32, name="epoch_avg_loss_ph")
            self.epoch_max_pos_ph = tf_v1.placeholder(tf.float32, name="epoch_max_pos_ph")
            self.epoch_eps_ph = tf_v1.placeholder(tf.float32, name="epoch_eps_ph")
            summary_dict = {"epoch_reward":self.epoch_reward_ph, "epoch_avg_loss":self.epoch_avg_loss_ph, 
                            "epoch_max_pos":self.epoch_max_pos_ph, "epoch_eps":self.epoch_eps_ph}
            summaries = [tf_v1.summary.scalar(summ_name, placeholder) for summ_name, placeholder in summary_dict.items()]
        return tf_v1.summary.merge(summaries)

    def get_summaries(self, feed_dict):
        """
        Executes summary op
        args:
            feed_dict (dict) : Feed dictionary as {self.epoch_reward_ph:epoch_reward,..., self.epoch_eps_ph:epoch_eps}
        returns:
            tf_v1.Summary: Summary protocol buffer 
        """
        summaries = self.sess.run(self.summaries, feed_dict=feed_dict)
        return summaries

    def calculate_eps(self, epoch, explore_ratio=0.25, explore_exploit_interval=20):
        """
        Return epsilon value
        args:
            epoch (int) : Training epoch value
            explore_ratio (float) : Fraction of explore_exploit_interval used to explore
            # TODO: Better description
            explore_exploit_interval (int) : Number of epochs within which the agent explores and exploits it's experience
        returns:
            float : Epsilon value
        """
        eps = 0.0
        # Explore or exploit
        explore = (epoch%explore_exploit_interval)<(explore_ratio*explore_exploit_interval)
        if explore:
            max_eps, min_eps = self.eps_limits
            n = int((epoch+1)/explore_exploit_interval)
            eps = max(max_eps*self.eps_decay**n, min_eps)
        return eps

    def _choose_action(self, state):
        """
        Return action value for given state
        args:
            state (gym.spaces.Box) : Box object as (position, velocity)
        returns:
            int : Action index
        """
        if np.random.random() < self.eps:
            return self.env.action_space.sample()
        else:
            q_sa_values = self.local_estimator.predict_one(self.sess, state)
            return np.argmax(q_sa_values)

    def train_models(self):
        """
        Trains the local estimator using sampled batch
        args:
            Nothing
        returns:
            float : Reduced average loss of the sampled batch
        """
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # Predict Q-values for current and next states
        q_sa_t0 = self.local_estimator.predict_batch(self.sess, states)
        q_sa_t1 = self.target_estimator.predict_batch(self.sess, next_states)
        # Empty numpy arrays for input and output batches
        x = np.zeros((len(states), *self.state_shape))
        y = np.zeros((len(actions), self.num_actions))
        for t, (state, action, reward, done) in enumerate(zip(states, actions, rewards, dones)):
            current_q, future_q = q_sa_t0[t], q_sa_t1[t]          # get the Q-values for all actions in current state
            if done:
                current_q[action] = (1-self.lr)*current_q[action] + self.lr*reward
                #current_q[action] = reward
            else:
                # Q(s,a) = (1-learning_rate)*Q(s,a) + learning_rate*(reward + discount_rate*Qmax(s+1,a+1))
                current_q[action] = (1-self.lr)*current_q[action] + self.lr*(reward + self.df*np.amax(future_q))
            x[t] = state
            y[t] = current_q
        batch_avg_loss = self.local_estimator.update(self.sess, x, y)
        return batch_avg_loss

    def transfer_weights(self):
        """
        Transfers partially or completely the weights from local estimator to target estimator
        """
        if self.weights_updater is None:
            print("Cannot transfer weights when target estimator is None!")
            return
        self.sess.run(self.weights_updater)

    def _run_once(self, epoch, target_update_steps=1000, explore_ratio=0.25, explore_exploit_interval=20, training=True):
        """
        Runs the simulation for one epoch
        args:
            epoch (int) : Epoch index
            target_update_steps (int) : Global step interval to update weights of target estimator
            explore_ratio (float) : Fraction of explore_exploit_interval used to explore
            explore_exploit_interval (int) : Number of epochs within which the agent explores and exploits it's experience
            training (bool) : Training or testing the model
        returns:
            (float, float, float) : Mean loss, total reward and maximum position of the current epoch
        """
        losses = []
        tot_reward = 0
        max_pos = self.env.observation_space.low[0]
        done = False
        state = self.env.reset()
        self.eps = self.calculate_eps(epoch, explore_ratio, explore_exploit_interval) if training else 0.0
        while not done:
            action = self._choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            if self.render: 
                self.env.render()
            self.replay_buffer.add(state, action, reward, next_state, done)
            if training:
                loss = self.train_models()
                losses.append(loss)
            tot_reward += reward
            max_pos = max(max_pos, next_state[0])
            state = next_state
            if training and self.ddqn and (not self.steps%target_update_steps):
                self.transfer_weights()
            self.steps += 1
        mean_loss = np.mean(losses) if losses else 0.0
        return mean_loss, tot_reward, max_pos

    def run(self, epochs, goal_trials=100, goal_reward=-110.0, explore_ratio=0.25, explore_exploit_interval=20, 
            display_every=100, target_update_steps=1000, return_result=True, training=True):
        """
        Runs the simulation for several epochs
        args:
            epochs (int) : Total number of epochs
            goal_trials (int) : Consecutive trials to calculate goal reward
            goal_reward (float) : Minimum average reward for goal_trials to succeed
            explore_ratio (float) : Fraction of explore_exploit_interval used to explore
            explore_exploit_interval (int) : Number of epochs within which the agent explores and exploits it's experience
            display_every (int) : Epoch interval to display the epoch information
            target_update_steps (int) : Global step interval to update weights of target estimator
            return_result (bool) : If true, store epoch summaries and returns them (to plot later in matplotlib)
            training (bool) : Training or testing the model
        returns:
            (list, list, list, list, tuple) : Losses, rewards, max positions and goal summary (number of goals achieved, first goal, max goal)
        """
        # Lists to store summaries
        losses = []
        rewards = []
        epsilons = []
        max_positions = []
        # Goal variables
        achieved_goals = 0
        first_goal = (None, None)
        max_mean_reward = (-1, -1000)
        assert goal_trials <= epochs, "Number of epochs must be atleast the number of goal trials!"
        print("Goal: Get average reward of {:.2f} over {} consecutive trials!".format(goal_reward, goal_trials))
        print("{} agent. Please be patient...".format("Training" if training else "Testing"), end="\r")
        t0 = time()
        run_args = (target_update_steps, explore_ratio, explore_exploit_interval, training)
        for epoch in range(1, epochs+1):
            mean_loss, tot_reward, max_pos = self._run_once(epoch, *run_args)
            if not epoch%display_every:                                     # Display epoch information
                t = time()-t0                                               # Measure time difference from previous display
                print_args = (epoch, mean_loss, tot_reward, max_pos, t)
                print("Epoch: {}, mean_loss: {:.4f}, total_reward: {}, max_pos: {:.4f}, in {:.4f} secs".format(*print_args))
                print("{} agent. Please be patient...".format("Training" if training else "Testing"), end="\r")
                t0 = time()
            rewards.append(tot_reward)
            # Store epoch information if user wants to plot them in matplotlib later
            if return_result:
                losses.append(mean_loss)
                max_positions.append(max_pos)
                epsilons.append(self.eps)
            # Check if the agent achieved required mean goal reward
            if len(rewards) >= goal_trials:
                mean_reward = np.mean(rewards[-goal_trials:])
                solved = (mean_reward >= goal_reward)
                if solved:
                    if first_goal[0] is None:
                        first_goal = (epoch, mean_reward)
                    achieved_goals += 1
                if mean_reward > max_mean_reward[1]:
                    max_mean_reward = (epoch, mean_reward)
            # Log summaries to display in Tensorboard
            if self.summ_writer is not None:
                feed_dict = {self.epoch_reward_ph:tot_reward, self.epoch_avg_loss_ph:mean_loss, self.epoch_max_pos_ph:max_pos, self.epoch_eps_ph:self.eps}
                summary = self.get_summaries(feed_dict)
                self.summ_writer.add_summary(summary, epoch)
        print("{:<50}".format(""), end="\r")
        # Prepare goal summary
        goal_summary = (achieved_goals, first_goal, max_mean_reward)
        return losses, rewards, max_positions, epsilons, goal_summary

    def save_model(self, chkpt_dir):
        """
        Saves variables from the session
        args:
            chkpt_dir (str) : Destination directory to store variables (as checkpoint)
        """
        self.saver.save(self.sess, chkpt_dir)

    def restore_model(self, chkpt_dir):
        """
        Restores varibles to the session
        args:
            chkpt_dir (str) : Source directory to restore variables (as checkpoint) 
        """
        self.saver.restore(self.sess, chkpt_dir)

    def log(self, logger, model_name, parameter_dict, goal_summary):
        """
        Logs parameters and goal summary
        args:
            logger (logging.Logger) : Logger object
            model_name (str) : Name of model
            parameter_dict (dict) : Dictionary with parameters to log
            goal_summary (tuple) : Tuple with goal summary
        """
        parameter_str = dict2str(parameter_dict)
        logger.debug("{} - {}".format(model_name, parameter_str))
        num_goals, first_goal, max_goal = goal_summary
        first_goal_epoch, first_goal_reward = first_goal 
        max_goal_epoch, max_goal_reward = max_goal
        logger.info("Goals achieved: {:<20}".format(num_goals))
        if num_goals:
            logger.info("First goal achieved: {:.2f} mean reward at {} epoch.".format(first_goal_reward, first_goal_epoch))
        logger.info("Max goal achieved: {:.2f} mean reward at {} epoch.\n".format(max_goal_reward, max_goal_epoch))
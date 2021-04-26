
# Reinforcement Learning - OpenAI gym
This repository contains the solutions for the OpenAI gym environments using different Deep Reinforcement Learning algorithms.  
**Objective:** For the default OpenAI Gym environments, their goals are to achieve a certain average threshold reward value for a consecutive number of trials (eposides) as available [here](https://github.com/openai/gym/wiki/Table-of-environments). For the environments other than that provided by the OpenAI Gym, their goal reward is set to `0` and number of trials to `1` by default. 

|    |    |  
| ------------- | ------------- |  
| <img src="assets/Images/CartPoleV0_Sarsa.gif" width="350" height="200" title="CartPole-v0 using SARSA algorithm"/>  | <img src="assets/Images/LunarLanderV2_DDQN.gif" width="350" height="200" title="LunarLander-v2 using DDQN algorithm"/>  |  
| <img src="assets/Images/MountainCarV0_DDQN.gif" width="350" height="200" title="MountainCar-v0 using DDQN algorithm"/>  | <img src="assets/Images/BipedalWalkerHardcoreV3_SAC.gif" width="350" height="200" title="BipedalWalkerHardcore-v3 using SAC algorithm"/>  |


## Conda installation
1. Install [Conda](https://docs.anaconda.com/anaconda/install/linux/)
2. Clone this repository (let's say to ${SRC_DIR})
3. Create and activate conda environment with following command  
```shell
cd ${SRC_DIR}  
conda env create -f environment.yml    
conda activate open_ai_gym
```

## Supported RL environments
- [OpenAI gym](https://gym.openai.com/envs)
  - Classic control
  - Box2D (except `CarRacing-v0` which consumes all memory as its state space consists of 96x96 RGB images)
  - Mujoco (needs activation as described [here](https://github.com/openai/mujoco-py))
  - Robotics
- [PyBullet](https://pybullet.org/wordpress/)
- [Highway-env](https://github.com/eleurent/highway-env) 

## Implemented RL algorithms
Following model-free Deep RL algorithms are available:  

| Off-Policy | On-Policy |  
| ------------- | ------------- |  
| DQN  | SARSA |  
| DDQN | REINFORCE|  
| DDPG | A2C |  
| SAC  |   |  


## Usage:
## Training the agent
- Create a YAML config file (let's say `sac_experiment.yaml`) 
```YAML
LunarLander-v2: &base_config
  env_name: LunarLander-v2
  epochs: 1000
  record_interval: 10
  render: False
  training: True
  summary_dir: summaries/classic_control
  algo:
    name: ddqn
    kwargs:
      tau: 0.005
      update_interval: 10
      num_gradient_steps: auto
      num_init_exp_samples: 2500
      max_init_exp_timestep: auto
      batch_size_kwargs:
        type: ConstantScheduler
        value: 32
      gamma_kwargs:
        type: ConstantScheduler
        value: 0.99
      lr_kwargs:
        type: ConstantScheduler
        value: 0.0001
  policy:
    name: greedy_epsilon
    kwargs:
      eps_kwargs:
        type: ExpScheduler
        decay_rate: 0.01
        update_step: 20
        clip_range: [0.01, 0.5]
```
[comment]: <> (Organise attributes and their descritions in a table)
- The valid parameters for the YAML config file are as follows:
    * `env_name`: (str) Name of the [OpenAI gym](https://github.com/openai/gym/wiki/Table-of-environments) / [PyBullet](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#) environment
    * `epochs`:  (int) Number of training/testing epochs. Defaults to **1000**. 
      > The number of time-steps per epoch depends on the environment. 
    * `render`: (bool) Option to render each epoch on the display. Defaults to **False**.
      > PyBullet environments cannot be rendered. Therefore, please look at their recorded videos for the rendered simualtions.
    * `record_interval` : (int) Interval (in terms of epoch) to record and save the given epoch as mp4 video. Defaults to **10**. 
      > For OpenAI Gym envrionments, recording videos also renders the recorded epoch on the display.
    * `algo`:
      * `name`: (str) Name of one of the supported algorithms from [here](/src/Algorithm) in *snake_case* naming convention.
      * `kwargs` : (dict) Arguments of the given algorithm as key-value pairs. Supported arguments for each algorithm can be found [here](src/config.py).  
    * `policy`:
      * `name`: (str) Name of one of the supported policies from [here](/src/Policy) in *snake_case* naming convention.
      * `kwargs`: (dict) Arguments of the given policy as key-value pairs. Supported arguments for each policy can be found [here](src/config.py).
    * `load_model`: (str) Path to the directory where a pretrained model is saved as a checkpoint. The weights of this model is restored to the current model.
      > Since only the weights are restored from the pretrained model, it is important that the source and target neural networks have the same architecture.
- Enter the following command:  
```shell
python train.py sac_experiment.yaml
```
The above command will train the agent on the `LunarLander-v2` environments using `DDQN` algorithm with `Greedy Epsilon` policy.

***
## Testing  the agent
- Create a YAML config similar to the one above (let's say `sac_experiment.yaml`) where `training` is set to `False` and 
  `load_model` is the directory path where the trained model's checkpoint is located (something like `summaries/box2d/LunarLander-v2-ddqn-greedy_epsilon-DD.MM.YYYY_mm.ss/model`).
- Enter the following command:
```shell
python train.py sac_experiment.yaml
```

***
## Summary information
- Track the summary in real-time with tensorboard using the command.  
```shell
tensorboard --host localhost --logdir summaries
```
The respective summary directory contains following files and directories:
- **model**: tensorboard summary and trained models (as checkpoints).
- **videos**: recorded videos if `--record_interval` argument was passed while training or testing the model
- **goal_info.yaml**: YAML file with given goal information: number of goals achieved, epoch and reward values for the first goal and max reward.
- **config.yaml**: YAML file with parameters used to train the given model.

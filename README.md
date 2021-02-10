
# Reinforcement Learning - OpenAI gym
Repository to solve the OpenAI gym environments using different Reinforcement Learning algorithms. The goal in each environment is to achieve a certain average threshold reward value for a consecutive 100 trials (epochs). These information about the environments are available in this OpenAI gym wiki [here](https://github.com/openai/gym/wiki/Table-of-environments).

|    |    |  
| ------------- | ------------- |  
| <img src="assets/Images/CartPoleV0_Sarsa.gif" width="350" height="200" title="CartPole-v0 using SARSA algorithm"/>  | <img src="assets/Images/LunarLanderV2_DDQN.gif" width="350" height="200" title="LunarLander-v2 using DDQN algorithm"/>  |
| <img src="assets/Images/MountainCarV0_DDQN.gif" width="350" height="200" title="MountainCar-v0 using DDQN algorithm"/>  | <img src="assets/Images/BipedalWalkerV2_DDPG.gif" width="350" height="200" title="BipedalWalker-v2 using DDPG algorithm"/>  |
 
# Requirements:
- OS: Windows 10/Ubuntu 18.04
- Python 3.7


 ## Conda installation
 1. Install [Conda](https://docs.anaconda.com/anaconda/install/linux/)
 2. Clone this repository (let's say to ${SRC_DIR})
3. Create and activate conda environment with following command  
```shell
cd ${SRC_DIR}  
conda env create -f environment.yml    
conda activate open_ai_gym
```

## Usage:
## Training the agent
- Create a YAML config file (let's say `sac_experiment.yaml`) 
```YAML
SAC_BipedalWalker:
  env_name: BipedalWalker-v3
  epochs: 1000
  record_interval: 10
  training: True
  render: True
  algo: 
    name: sac
    kwargs:
      lr: 0.003
      gamma: 0.99
      batch_size: 256
      ...
  policy:
    name: gaussian
    kwargs:
      lr: 0.003
      ...

SAC_Walker2DBulletEnv:
  env_name: Walker2DBulletEnv-v0
  epochs: 1000
  record_interval: 10
  training: True
  render: False
  algo: 
    name: sac
    kwargs:
      lr: 0.003
      gamma: 0.99
      batch_size: 256
      ...
  policy:
    name: gaussian
    kwargs:
      lr: 0.003
      ...
```
- The valid parameters for the YAML config file are as follows:
    * `env_name` (str): Name of the [OpenAI gym](https://github.com/openai/gym/wiki/Table-of-environments) / [PyBullet](https://github.com/benelot/pybullet-gym) environment
    * `epochs` (int): Number of training/testing epochs. Defaults to **1000**. 
      > The number of time-steps per epoch depends on the environment. 
    * `render` (bool): Option to render each epoch on the display. Defaults to **False**.
      > PyBullet environments cannot be rendered. Therefore, please look at their recorded videos for the rendered simualtions.
    * `record_interval` (int): Interval (in terms of epoch) to record and save the given epoch as mp4 video. Defaults to **10**. This also renders the recorded epoch on the display.
    * `algo`:
      * `name` (str): Name of one of the supported algorithms from [here](/src/Algorithm) in *snake_case* naming convention.
      * `kwargs` (dict): Arguments of the given algorithm as key-value pairs. Supported arguments for each algorithm can be found [here](src/config.py).  
    * `policy`:
      * `name` (str): Name of one of the supported policies from [here](/src/Policy) in *snake_case* naming convention.
      * `kwargs` (dict): Arguments of the given policy as key-value pairs. Supported arguments for each policy can be found [here](src/config.py).
    * `load_model` (str): Path to the directory where a pretrained model is saved as a checkpoint. The weights of this model is restored to the current model.
      > Since only the weights are restored from the pretrained model, it is important that the source and destination networks have same architecture.
- Enter the following command (use --help to see all arguments):  
```shell
python rl.py sac_experiment.yaml
```
The above command will train the agent on two different environments (`BipedalWalker-v3` and `Walker2DBulletEnv`) sequentially using `SAC` algorithm.

***
## Testing  the agent
- Create a YAML config similar to the one above (let's say `sac_experiment.yaml`) where `training` is set to False and 
  `load_model` is the directory path where the trained model's checkpoint is located.
- Enter the following command:
```shell
python rl.py sac_experiment.yaml
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
- **goal_info.json**: JSON file with given goal information: number of goals achieved, epoch and reward values for the first goal and max reward.
- **config.json**: JSON file with parameters used to train the given model.

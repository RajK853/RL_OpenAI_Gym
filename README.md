
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
  render: False
  training: True
  load_model: null
  algo: 
    name: sac
    kwargs:
      batch_size: 100
      num_init_exp_samples: 10000
      tau: 0.001
      update_interval: 1
      num_q_nets: 3
      clip_norm: 5.0
      auto_ent: True
      target_entropy: auto
      init_log_alpha: 0.0
      gamma_kwargs:
        type: ConstantScheduler
        value: 0.99
      q_lr_kwargs:
        type: ConstantScheduler
        value: 0.0001
      alpha_lr_kwargs:
        type: ConstantScheduler
        value: 0.0001
  policy:
    name: gaussian
    kwargs:
      clip_norm: 5.0
      learn_std: True
      mu_range: [-2.0, 2.0]
      log_std_range: [-20, 0.1]
      lr_kwargs:
        type: ConstantScheduler
        value: 0.0001
```
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
      > Since only the weights are restored from the pretrained model, it is important that the source and destination networks have same architecture.
- Enter the following command (use --help to see all arguments):  
```shell
python train.py sac_experiment.yaml
```
The above command will train the agent on the `BipedalWalker-v3` environments using `SAC` algorithm.

***
## Testing  the agent
- Create a YAML config similar to the one above (let's say `sac_experiment.yaml`) where `training` is set to False and 
  `load_model` is the directory path where the trained model's checkpoint is located.
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

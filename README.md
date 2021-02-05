
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
Read the documentation provided in [here](run/README.md) for training and testing procedures.
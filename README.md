
# Reinforcement Learning - OpenAI Gym
Repository to solve the OpenAI Gym RL [environments](https://github.com/openai/gym/wiki/Table-of-environments) using different Reinforcement Learning algorithms.

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

|    |   |  
| ------------- | ------------- |  
| <img src="assets/CartPoleV0_Sarsa.gif" width="260" height="300" title="CartPole-v0 using SARSA algorithm"/>  | <img src="assets/LunarLanderV2_DDQN.gif" width="300" height="200" title="LunarLander-v2 using DDQN algorithm"/>  |
 
## TODO:
- ~~Complete code for Gaussian policy for continuous action space.~~
- Add support for different replay buffers; PER, HER, etc.
- Add support for different algorithms; ~~SARSA~~, ~~DDPG~~, SAC, etc.
- Include preprocessors for neural networks. For instance a CNN network to process state images.
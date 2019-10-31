
# Reinforcement Learning - OpenAI Gym
 Respository with solutions for OpenAI Gym RL environments.

 # Requirements:
 - Python 3.7
 - OS: Windows 10/Ubuntu 18.04

 ## Available solutions:
 - MountainCar-v0
 - LunarLander-v2
 - CartPole-v0

 ## Training the agent
 - In the **config** directory, edit the parameter values for required environment.
 - Open a command terminal in the git-cloned root directory.
 - Enter the following command (use --help to see all parsable arguments):  
 ```python -m run.main --env_name "LunarLander-v2" --epochs 1000 --record_interval 100```
 - Track the summary in real-time with tensorboard using the command  
   ```tensorboard --host localhost --logdir "summaries\ENV_SUMM_DIR"```
 - The respective summary directory contains following directories:
 	- **log**: log information about trained models
 	- **Model num**: tensorboard summary and trained models (as checkpoints)
 	- **videos**: recorded videos if *--record_interval* argument was passed while training or testing the model
 	
 ## Testing  the agent
 - In the **config** directory, edit the parameter values for required environment.
 - Open a command terminal in the git-cloned root directory.
 - Enter the following command:  
  ```python -m run.main --env_name "LunarLander-v2" --test_model_chkpt "MODEL_CHECKPOINT_ADDRESS/model.chkpt"```

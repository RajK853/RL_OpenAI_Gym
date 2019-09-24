
# Reinforcement Learning - OpenAI Gym
 Respository with solutions for OpenAI Gym RL environments.

 # Requirements:
 1. tensorflow-gpu 1.14
 2. gym
 3. matplotlib

 ## Available solutions:
 1. MountainCar-v0
 2. LunarLander-v2

 ## Training the agent
 1. In the **config** directory, edit the parameter values for required environment.
 2. Open a command terminal in the root directory.
 3. Enter ```python -m run.ENV_NAME``` (use --help to see all parsable arguments)
 4. Track the summary in real-time with tensorboard using the command ```tensorboard --host localhost --logdir "summaries\ENV_SUMM_DIR```
 5. The respective summary directory contains following directories:
 	- **log**: log information about trained models
 	- **Model num**: tensorboard summary and trained models (as checkpoints)
 	- **videos**: recorded videos if *--record_interval* argument was passed while training or testing the model
 	
 ## Testing  the agent
 1. In the **config** directory, edit the parameter values for required environment.
 2. Open a command terminal in the root directory.
 3. Enter ```python -m run.ENV_NAME --test_model_chkpt "MODEL_ADDRESS/model.chkpt"``` (use --help to see all parsable arguments)

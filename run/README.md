
 # Running the agent

 ## Training the agent
 - Open command terminal in the root directory (where you cloned the git repository).
 - Enter the following command (use --help to see all arguments):  
   ```python -m run --env_name "LunarLander-v2" --epochs 1000 --render True --display_interval 200 --record_interval 100 --algorithm "ddqn" --policy "greedy_epsilon" --goal_trials 100 --goal_reward 200```
 	
 ## Testing  the agent
 - Open command terminal in the root directory.
 - Along with the code to train the agent, provide directory with model checkpoint to `--test_mode_chkpt` argument.
 
 ## Summary information
  - Track the summary in real-time with tensorboard using the command.  
   ```tensorboard --host localhost --logdir "summaries"```
 - The respective summary directory contains following directories:
 	- **Model N**: tensorboard summary and trained models (as checkpoints)
 	- **videos**: recorded videos if `--record_interval` argument was passed while training or testing the model

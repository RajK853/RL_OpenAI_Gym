## Training the agent
- Open command terminal in the root directory (where you cloned the git repository).
- Enter the following command (use --help to see all arguments):  
```shell
python -m run --env_name "CartPole-v0" --epochs 1000 --render True --record_interval 100 --algorithm "ddqn" --policy "greedy_epsilon" --num_exec 1 
```
In the above command:
* `env_name` (str) : Name of the OpenAI gym environment. The list of valid environments can be found [here](https://github.com/openai/gym/wiki/Table-of-environments).
* `epochs` (int) : Number of training/testing epochs. Defaults to **1000**. The number of time-steps per epoch depends on the environment. 
* `render` (boolean) : Option to render every epoch on the display. Defaults to **False**.
* `record_interval` (int) : Interval (in terms of epoch) to record and save the given epoch as mp4 video. Defaults to **10**. This also renders the recorded epoch on the display.
* `algorithm` (str) : Name of one of the supported algorithms from [here](../src/Algorithm). It follows *snake_case* naming convention.
* `policy` (str): Name of one of the supported policies from [here](../src/Policy). It follows *snake_case* naming convention without the prefix *-policy*.
* `num_exec` (int) : Number of execution of the entire training process with different random seeds. Defaults to **1**.

***
## Testing  the agent
- Open command terminal in the root directory.
- Along with the code to train the agent, provide directory with model checkpoint to `--test_mode_chkpt` argument.

***
## Summary information
- Track the summary in real-time with tensorboard using the command.  
```shell
tensorboard --host localhost --logdir summaries
```
The respective summary directory contains following directories:
- **Model N**: tensorboard summary and trained models (as checkpoints).
- **videos**: recorded videos if `--record_interval` argument was passed while training or testing the model

import os
import sys
import subprocess
import streamlit as st
from pyvirtualdisplay import Display

from test import setup

def rollout(env, policy):
  done = False
  state = env.reset()
  while not done:
      action = policy(state)
      next_state, reward, done, info = env.step(action)
      state = next_state
      state_img = env.render(mode="rgb_array")
      yield state_img

def execute(env_name, model_path, epochs, dump_path, include):
  with Display(visible=False) as disp:
    env, policy = setup(env_name, model_path, dump_path, include)
    with st.empty():
      for epoch in range(1, epochs+1):
        for img in rollout(env, policy):
          st.image(img, caption=f"Epoch {epoch}")


@st.cache
def get_os(platform):
  if platform.startswith("linux"):
    return "linux"
  elif platform == "darwin":
    return "osx"
  return "windows"


@st.cache
def read_file(name, *args):
  with open(name, *args) as fp:
    return fp.read()


# Constants
OS = get_os(sys.platform)
sbar = st.sidebar

# Streamlit components
env_name = sbar.text_input("Environment ID", key="env_name_i")
model_path = sbar.text_input("Model path", key="model_path_i")
epochs = sbar.number_input("Epochs", 1, 10, key="epoch_i")
include = sbar.text_input("Additional imports", key="include_i")
start_btn = sbar.button("Start Testing", key="start_btn_i")

video_path = os.path.join("temp", env_name)

# Descriptions
st.title("OpenAI Gym - Reinforcement Learning Algorithms")

st.write("""
Use this application to test trained models in their respective RL environment.
This is a web interface that executes the `test.py` script underneath to record
the rollouts as `mp4` videos.  


Parameter information:
- **Environment ID**: Name of the RL environment.  
  > Supported environments: [OpenAI gym](https://gym.openai.com/envs), [Mujoco](https://github.com/openai/mujoco-py), [PyBullet](https://pybullet.org/wordpress/) and [Highway-env](https://github.com/eleurent/highway-env) 
- **Model path**: Path to the trained `.tf` or `.h5` model
  > Samples models are present in the `models` directory
- **Epochs**: Number of testing epochs
- **Additional imports**: Name of the Python package to import to load the above RL environment.  
  > For instance, to load `PyBullet` environments, write `pybullet_envs` here.
""")

st.header("Rollout videos")

st.info(f"OS: {OS.capitalize()}")

if start_btn:
  """cmd = []
  if OS == "linux":
    cmd.extend(["xvfb-run", "-a"])
  cmd.extend([
    "python", "test.py", 
    "--env_name", env_name, 
    "--epochs", str(epochs), 
    "--load_model", model_path, 
    "--dump_path", video_path,
  ])
  if include:
    cmd.extend(["--include", include])
  st.write("Executing the command:")
  st.code(' '.join(cmd), language="shell")
  st.info(f"Please be patient. \nTesting the policy in '{env_name}'.")
  subprocess.run(cmd)"""
  execute(env_name, model_path, epochs, video_path, include)

if os.path.exists(video_path):
  files = [os.path.join(video_path, filename) for filename in os.listdir(video_path) if filename.endswith(".mp4")]
  for i, filename in enumerate(files, start=1):
    st.subheader(f"Epoch {i}")
    video_bytes = read_file(filename, "rb")
    st.video(video_bytes)
else:
  st.error("No video found!")

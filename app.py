import os
import subprocess
import pandas as pd
import streamlit as st

sidebar = st.sidebar

@st.cache
def read_file(name, *args):
	with open(name, *args) as fp:
		return fp.read()

# Streamlit components
env_name = sidebar.text_input("Environment ID", key="env_name_i")
model_path = sidebar.text_input("Model path", key="model_path_i")
epochs = sidebar.number_input("Epochs", 1, 10, key="epoch_i")
include = sidebar.text_input("Additional imports", key="include_i")
start_btn = sidebar.button("Start Testing", key="start_btn_i")
apt_package = sidebar.text_input("Packages", key="apt_package_i")
apt_install_btn = sidebar.button("apt-install", key="apt_install_btn_i")
# Video dump path
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

if start_btn:
	cmd = ["python", "test.py", "--env_name", env_name, "--epochs", str(epochs), "--load_model", model_path, "--dump_path", video_path]
	if include:
		cmd.extend(["--include", include])
	st.write("Executing the command:")
	st.code(' '.join(cmd), language="shell")
	st.info(f"Please be patient. \nTesting the policy in '{env_name}'.")
	subprocess.run(cmd)

if apt_install_btn:
	if apt_package:
		st.info(f"Installing {apt_package}..")
		cmd = ["sudo" "apt-get", "install", apt_package]
		subprocess.run(cmd)
	else:
		st.error("Please enter the package to install!")

if os.path.exists(video_path):
	files = [os.path.join(video_path, filename) for filename in os.listdir(video_path) if filename.endswith(".mp4")]
	for i, filename in enumerate(files, start=1):
		st.subheader(f"Epoch {i}")
		video_bytes = read_file(filename, "rb")
		st.video(video_bytes)
else:
	st.error("No video found!")

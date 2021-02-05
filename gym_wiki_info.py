"""
This script downloads the OpenAI Gym information from their wiki at
'https://github.com/openai/gym/wiki/Table-of-environments' and saves it as a CSV file.
"""
import os
import urllib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


def fix_bipedal_walker_version(df):
    """
    Updates the BipedalWalker-v2 from the OpenAI Gym wiki to v3.
    Issue found in the BipedalWalker-v2 is mentioned here:
    https://github.com/ray-project/ray/issues/5001
    :param df: (DataFrame) DataFrame with OpenAI Gym env information
    :return: (DatFrame) Updated DataFrame
    """
    old_names = ("BipedalWalker-v2", "BipedalWalkerHardcore-v2")
    new_names = ("BipedalWalker-v3", "BipedalWalkerHardcore-v3")
    df["Environment Id"] = df["Environment Id"].replace(old_names, new_names)
    print(f"# Applied fix: Updated {old_names} to {new_names}!")
    return df


def download(url=r"https://github.com/openai/gym/wiki/Table-of-environments", dump_dir="assets"):
    client = urllib.request.urlopen(url)
    soup = BeautifulSoup(client, 'html.parser')
    print(f"# Loaded information from '{url}'")
    table = soup.find(name="table", attrs={"role": "table"})
    raw_data = [header.text.strip("\n").split("\n") for header in table.find_all("tr")]
    columns = raw_data.pop(0)
    raw_data = np.array(raw_data)
    data_dict = {col: raw_data[:, i] for i, col in enumerate(columns)}
    df = pd.DataFrame(data_dict)
    df = fix_bipedal_walker_version(df)
    os.makedirs(dump_dir, exist_ok=True)
    save_file = os.path.join(dump_dir, "env_info.csv")
    df.to_csv(save_file, index=False)
    print(f"# OpenAI Gym wiki info dumped as '{save_file}'")


if __name__ == "__main__":
    download(url=r"https://github.com/openai/gym/wiki/Table-of-environments", dump_dir="assets")

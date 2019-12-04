import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(os.path.abspath(""), "..")))
sys.path.insert(0, ROOT_DIR)

from .neural_network import NeuralNetwork
from .q_network import QNetwork
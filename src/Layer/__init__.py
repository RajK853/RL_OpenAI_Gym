import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(os.path.abspath(""), "..")))
sys.path.insert(0, ROOT_DIR)

from .NeuralNetwork import NeuralNetwork
from .QNetwork import QNetwork
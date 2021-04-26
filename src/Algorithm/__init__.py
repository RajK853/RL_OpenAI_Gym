# TODO: Find technique to register the algorithms instead of manually importing them here
from .base_algorithm import BaseAlgorithm
from .on_policy import OnPolicyAlgorithm
from .off_policy import OffPolicyAlgorithm
from .dqn import DQN
from .ddqn import DDQN
from .reinforce import Reinforce
from .a2c import A2C
from .sarsa import Sarsa
from .ddpg import DDPG
from .sac import SAC

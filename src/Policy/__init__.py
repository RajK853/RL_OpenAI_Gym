from .base_policy import BasePolicy
from .discrete import DiscretePolicy
from .gaussian import GaussianPolicy
from .greedy_epsilon import GreedyEpsilonPolicy
from .continuous import ContinuousPolicy

POLICIES = {"greedy_epsilon": GreedyEpsilonPolicy,
            "gaussian": GaussianPolicy,
            "discrete": DiscretePolicy,}


def get_policy(name, **kwargs):
    policy_func = POLICIES[name]
    return policy_func(**kwargs)

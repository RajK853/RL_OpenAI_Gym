import importlib
from dataclasses import dataclass
from class_registry import ClassRegistry

@dataclass
class registry:
	algorithm = ClassRegistry(attr_name="__new__")
	policy = ClassRegistry(attr_name="__new__")
	buffer = ClassRegistry(attr_name="__new__")


def load_algo(name, **kwargs):
	importlib.import_module(f"src.Algorithm.{name}")
	algo = registry.algorithm.get(name, **kwargs)
	return algo

def load_policy(name, **kwargs):
	importlib.import_module(f"src.Policy.{name}")
	policy = registry.policy.get(name, **kwargs)
	return policy

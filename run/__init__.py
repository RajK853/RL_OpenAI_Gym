import os
import sys

"""
Insert root dirctory in sys.path so that modules in src directory can be assessed
by all the modules in this directory
"""
ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(""), '..'))
sys.path.insert(0, ROOT_DIR)
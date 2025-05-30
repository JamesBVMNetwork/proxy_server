from yaml import load, Loader
import os

config_path = os.path.join(os.path.dirname(__file__), '8x4090.yaml')

CONFIG = load(open(config_path, "r"), Loader=Loader)
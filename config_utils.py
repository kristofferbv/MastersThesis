  
import yaml
import os

def load_config(path):
    if not os.path.exists(path):
        path = "../" + path

    with open(path, "r") as config:
        return yaml.full_load(config)
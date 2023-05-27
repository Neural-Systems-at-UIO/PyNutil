import json
from pathlib import Path
import os


def load_config() -> dict:
    """
    Loads the config file

    :return: the configuration file
    :rtype: dict
    """
    # returns a path to the config file assuming that it is in the same directory as this script
    path = str(Path(__file__).parent.parent.absolute()) + "/metadata/config.json"
    # open the config file
    with open(path, "r") as f:
        # load the config file
        config = json.load(f)
    # return the config file and path
    return config, path

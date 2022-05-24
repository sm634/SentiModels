# MyPy for Static Typing
from typing import List, Set, Dict, Tuple, Optional, Any, Iterable, Union

# PyPi Modules
import json
import os
from os.path import splitext
import toml
import yaml


class Config:
    def __init__(self) -> None:
        print(os.getcwd())
        self.config: dict = self.__getConfig()

    def get(self) -> dict:
        return self.config

    def __getConfig(self) -> dict:
        envVariables: os._Environ = os.environ
        if "PYTHON_ENV" in envVariables and "PYTHON_CONFIG_DIR" in envVariables:
            python_env = os.getenv("PYTHON_ENV")

            extension: str = str(splitext(python_env)[1])

            python_config_dir: str = str(os.getenv("PYTHON_CONFIG_DIR"))

            python_config_dir = python_config_dir.rstrip('/')

            file_path: str = f'{python_config_dir}/{python_env}'

            config: dict

            if extension == '.json':
                with open(file_path) as config_data:
                    config = json.load(config_data)
                    return config
            elif extension == '.toml':
                return toml.load(file_path)
            elif extension == '.yaml':
                with open(file_path) as config_data:
                    config = yaml.safe_load(config_data)
                    return config
            else:
                raise ConfigError(f"Config file extension is {extension}, file extension for config \
                                    file must be [.yaml, .toml or .json], ")
        else:
            raise ConfigError("PYTHON_ENV or/and PYTHON_CONFIG_DIR environment variables not declared \
                               for configuration file")


class ConfigError(Exception):
    pass


config = Config().get()

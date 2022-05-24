# Custom Modules
from .config import config

# MyPy for Static Typing
from typing import List, Set, Dict, Tuple, Optional, Any, Iterable, Union

# PyPi Modules
import logging


def getLogger() -> logging.Logger:

    logger_location = config["logging"]["logger_location"]

    logger_level = config["logging"]["logger_level"]

    loggers = logging.getLogger(__name__)

    loggers.setLevel(logger_level)

    if not len(loggers.handlers):  # Checking if handlers for job does not exist

        fh = logging.FileHandler(logger_location)

        fh.setLevel(logger_level)

        formatter = logging.Formatter('[%(asctime)s] [%(threadName)s] \
                    {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)

        loggers.addHandler(fh)

        ch = logging.StreamHandler()

        ch.setLevel(logger_level)

        ch.setFormatter(formatter)

        loggers.addHandler(ch)

    return loggers


logger: logging.Logger = getLogger()

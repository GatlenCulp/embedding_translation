"""Adds a bunch of random sugar for setup."""

import numpy as np
import rich.traceback

from utils.log_config import setup_logger


def setup(file_name: str) -> np.random.Generator:
    """Generally sets up project with nice sugar.

    Args:
        file_name: File name this is being run in
    """
    setup_logger(filename=file_name)
    rich.traceback.install()
    return np.random.default_rng(43)

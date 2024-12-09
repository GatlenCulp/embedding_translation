"""Adds a bunch of random sugar for setup."""

from src.utils.logging import setup_logger


def setup(file_name: str) -> None:
    """Generally sets up project with nice sugar.

    Args:
        file_name: File name this is being run in
    """
    setup_logger(filename=file_name)

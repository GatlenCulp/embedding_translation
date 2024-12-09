"""Configures logging for the entire project."""

from pathlib import Path

from loguru import logger
from rich.logging import RichHandler


PROJECT_ROOT = Path(__file__).parent.parent.parent

def setup_logger(filename: str) -> None:
    """Configure logger with file and console outputs.

    :param str filename: Name of the log file (without .log extension)
    """
    # Setup paths
    log_path = PROJECT_ROOT / "logs" / f"{filename}.log"
    log_path.parent.mkdir(exist_ok=True)

    # Remove default logger and add file + console logging
    logger.remove()
    logger.add(
        log_path,
        rotation="1 day",
        retention="1 week",
        level="INFO",
        format="[{time:YYYY-MM-DD HH:mm:ss}] {level} {message}",
    )
    logger.add(
        RichHandler(rich_tracebacks=True),
        level="INFO",
        format="{message}",
    )

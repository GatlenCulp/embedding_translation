"""Configures logging for the entire project."""

from pathlib import Path
from typing import ClassVar

from loguru import logger
from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.logging import RichHandler
from rich.theme import Theme


PROJECT_ROOT = Path(__file__).parent.parent.parent


class LogHighlighter(RegexHighlighter):
    """Highlight paths and common logging patterns."""

    highlights: ClassVar = [
        # Highlight any file paths (with any extension or directory structure)
        r"(?P<path>(?:[\w-]+/)*[\w.-]+(?:\.[a-zA-Z0-9]+)?)",  # <--- [CHANGED] Updated to match all paths
        # Highlight line numbers
        r"(?P<linenumber>:\d+)",
        # Highlight log levels
        r"(?P<level>INFO|SUCCESS|WARNING|ERROR|DEBUG)",
        # Highlight timestamps
        r"(?P<timestamp>\[\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\])",
    ]

    base_style = "log."


def setup_logger(filename: str) -> None:
    """Configure logger with file and console outputs.

    :param str filename: Name of the log file (without .log extension)
    """
    # Setup paths
    log_path = PROJECT_ROOT / "logs" / f"{filename}.log"
    log_path.parent.mkdir(exist_ok=True)

    # Remove default logger and add file logging
    logger.remove()
    logger.add(
        log_path,
        rotation="1 day",
        retention="1 week",
        level="INFO",
        format="[{time:YYYY-MM-DD HH:mm:ss}] {level} {message}",
    )

    # Update console logging with custom highlighting
    theme = Theme(
        {
            "log.path": "bold cyan",
            "log.linenumber": "bold yellow",
            "log.level": "bold green",  # <--- [CHANGED] Made levels more visible
            "log.timestamp": "dim blue",  # <--- [NEW] Added timestamp style
        }
    )

    console = Console(theme=theme, highlighter=LogHighlighter())

    logger.add(
        RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_path=True,
            show_time=True,
            keywords=["error", "warning", "debug", "info"],
        ),
        level="INFO",
        format="[{file}:{line}] {message}",
    )


if __name__ == "__main__":
    # Test different types of log messages
    setup_logger("test_logging")

    logger.info("Starting logging test...")

    # Test path highlighting
    logger.info("Loading configuration from /path/to/config.json")
    logger.info("Processing data from C:\\Users\\test\\data.csv")

    # Test line number highlighting
    logger.error("Failed to process file at line 42")

    # Test log level highlighting
    logger.debug("DEBUG message test")
    logger.info("INFO message test")
    logger.warning("WARNING message test")
    logger.error("ERROR message test")

    # Test combined highlighting
    logger.error("Error in file /path/to/script.py at line 123: Failed to process data")

    logger.info("Logging test completed!")

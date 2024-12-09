"""Dumps any given datastructure to a json file.

Description: Provides functionality to serialize and save Python objects to JSON files
with configurable settings and proper type handling. Includes support for common
Python types like datetime, Path objects, and NumPy arrays.
"""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.utils.general_setup import setup


setup("anal_dump")


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for handling special Python types.

    :param Any obj: Object to serialize
    :return: JSON serializable representation of the object
    :rtype: Any
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _save_json(
    data: Any,
    filepath: Path,
    indent: int = 2,
    serializer: Callable = _json_serializer,
) -> Path:
    """Save data as a JSON file.

    :param Any data: Data to save
    :param Path filepath: Full path including filename and .json extension
    :param int indent: Number of spaces for indentation
    :param Callable serializer: Custom JSON serializer function
    :return: Path to the saved JSON file
    :rtype: Path
    """
    logger.info(f"Saving JSON to {filepath}")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=indent,
                default=serializer,
                ensure_ascii=False,
            )
        logger.success(f"Successfully saved JSON to {filepath}")

    except Exception as e:
        logger.error(f"Failed to save JSON: {e!s}")
        raise

    return filepath


def anal_dump(
    data: Any,
    filename: str | Path,
    output_dir: str | Path = "data/dumps",
    indent: int = 2,
) -> Path:
    """Save any data structure as a JSON file.

    :param Any data: Data structure to save
    :param str | Path filename: Name of the file without extension
    :param str | Path output_dir: Directory to save the file in
    :param int indent: Number of spaces for indentation
    :return: Path to the saved JSON file
    :rtype: Path
    """
    # Convert paths to Path objects
    output_dir = Path(output_dir)
    filename = Path(filename).stem  # Get filename without extension

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate file path
    json_path = output_dir / f"{filename}.json"

    # Save JSON file
    return _save_json(data, json_path, indent=indent)


def main() -> None:
    """Run example data dumping."""
    logger.info("Starting example data dump")

    # Example data structure with various types
    example_data = {
        "string": "Hello, World!",
        "number": 42,
        "float": 3.14159,
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
        "datetime": datetime.now(),
        "path": Path("some/path"),
        "numpy_array": np.array([1, 2, 3]),
        "numpy_int": np.int64(42),
        "numpy_float": np.float64(3.14159),
    }

    # Save the example data
    json_path = anal_dump(
        data=example_data,
        filename="example_dump",
    )
    logger.info(f"Example data dumped to {json_path}")


if __name__ == "__main__":
    main()

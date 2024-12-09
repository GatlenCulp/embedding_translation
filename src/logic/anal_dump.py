"""Dumps any given datastructure to a json file using Pydantic.

Description: Provides functionality to serialize and save Python objects to JSON files
using Pydantic for validation and serialization. Includes support for common
Python types like datetime, Path objects, and NumPy arrays.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_serializer

from src.utils.general_setup import setup


setup("anal_dump")


class DataFile(BaseModel):
    """Pydantic model for data serialization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pydantic_schema: str = Field(description="The pydantic schema that hte data is saved in")

    name: str = Field(description="Name of this piece of data")

    # Metadata fields
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="UTC timestamp when the dump was created"
    )
    version: str = Field(default="1.0.0", description="Version of the data format")
    description: str | None = Field(
        default=None, description="Optional description of the data dump"
    )

    # Original data field
    data: Any = Field(description="Data to be serialized")

    additional_fields: dict[str, Any] | None = None

    @field_serializer("data")
    def serialize_data(self, v: Any) -> Any:
        """Custom serializer for handling special Python types."""
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.int64, np.int32)):
            return int(v)
        if isinstance(v, (np.float64, np.float32)):
            return float(v)
        return v

    @field_serializer("created_at")
    def serialize_datetime(self, v: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return v.isoformat()


def anal_dump(
    data: BaseModel,
    filename: str | Path,
    output_dir: str | Path = "data/anal",
    indent: int = 2,
) -> Path:
    """Save a Pydantic model as a JSON file."""
    # Convert paths to Path objects
    output_dir = Path(output_dir)
    filename = Path(filename).stem

    # Generate file path
    json_dir = output_dir
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"{filename}.json"

    # Convert the input model to a dictionary first
    data_dict = {"data": data.model_dump()}
    data = DataFile(name=filename, pydantic_schema=data.__class__.__name__, **data_dict)

    try:
        # Save JSON file directly from the Pydantic model
        logger.info(f"Saving JSON to {json_path}")
        with json_path.open("w", encoding="utf-8") as f:
            f.write(data.model_dump_json(indent=indent))
        logger.success(f"Successfully saved JSON to {json_path}")

    except Exception as e:
        logger.error(f"Failed to save JSON: {e!s}")
        raise

    return json_path


def main() -> None:
    """Run example data dump."""
    logger.info("Starting example data dump")

    # Example Pydantic model
    class ExampleModel(BaseModel):
        """Example Pydantic model for demonstration."""

        text_str: str
        count: int
        decimal: float
        items: list
        mapping: dict
        datetime: datetime
        path: Path

    # Create example data with the Pydantic model
    example_data = ExampleModel(
        text_str="Hello, World!",
        count=42,
        decimal=3.14159,
        items=[1, 2, 3],
        mapping={"a": 1, "b": 2},
        datetime=datetime.now(),
        path=Path("some/path"),
    )

    # Save the example data
    json_path = anal_dump(
        data=example_data,
        filename="example_dump",
    )
    logger.info(f"Example data dumped to {json_path}")


if __name__ == "__main__":
    main()

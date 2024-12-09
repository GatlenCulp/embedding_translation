"""Aggregate statistics across all the stitches."""

from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import computed_field
from pydantic import model_validator

from src.schema.training_schemas import EmbeddingDatasetInformation, StitchSummary


class AggregateStatistics(BaseModel):
    """Aggregate statistics across all the stitches. Can be rednered into table."""

    # Input
    summaries: list[StitchSummary] = Field(description="Literally every training run as input.")

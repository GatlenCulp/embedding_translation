"""Aggregate statistics across all the stitches."""

from pydantic import BaseModel
from pydantic import Field

from src.schema.training_schemas import StitchSummary


class AggregateStatistics(BaseModel):
    """Aggregate statistics across all the stitches. Can be rendered into table.

    NOTE: Probably want to use some sort of order between the final results
    """

    # Input
    summaries: list[StitchSummary] = Field(description="Literally every training run as input.")



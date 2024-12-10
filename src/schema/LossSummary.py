"""Table of comparative MSE loss."""

from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import computed_field

from src.schema.training_schemas import EmbeddingDatasetInformation


class LossSummary(BaseModel):
    """Table of MSE loss.

    MSE Comparison, used to make a table
    Each row represents the embedding space
    Each column represents the stitch model to convert to the embedding space
    Each entry is some value
    """

    # TODO: Write this

    # GAH I should make a function that generally takes in some 2D table of
    # `StitchSummaries` and performs an op on it to make a plotly table.

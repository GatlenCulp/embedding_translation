"""Represents the comparative statistics for semantic search.

Exists to see whether semantic search results are preserved by stitching.

Uses (kNNs) with stitch and source embeddings to compare accuracy.
"""

from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import computed_field

from src.schema.training_schemas import EmbeddingDatasetInformation


class SemanticSearchEvaluation(BaseModel):
    """Represents the results from performing kNN on test dataset using converted training dataset."""

    # Data Source
    training_dataset: EmbeddingDatasetInformation = Field(
        description="stitched or non-stitched dataset containing the labels."
    )
    test_dataset: EmbeddingDatasetInformation = Field(
        description="stitched or non-stitched dataset to label."
    )

    # Process
    distance_function: Literal["euclidean"] = Field(description="Name of distance function used")
    # NOTE: For now we just use one/the nearest
    k: int = Field(description="Number of neighbors the test data was labled with", gt=0)

    # Results
    nearest_neighbors: dict[str | int, dict[str | int, float]] = Field(
        description="Dictionary mapping test_dataset record_ids to their nearest training_dataset record_ids and the distance to each"
    )
    labels: dict[str | int, str | int] = Field(
        description="Dictionary of test_dataset record_ids to their labels"
    )

    @computed_field
    @property
    def sample_size(self) -> int:
        """Get the number of samples based on record_ids length."""
        return len(self.record_ids)


class SemanticSearchPairwiseEvaluation(BaseModel):
    """Represents the comparative statistics for semantic search."""

    target_semantic_search: SemanticSearchEvaluation = Field(
        description="The target semantic search"
    )
    # Okay I should really just build stuff.

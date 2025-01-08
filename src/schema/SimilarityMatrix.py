"""Contains schema for similarity matrices."""

from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import computed_field
from pydantic import model_validator

from src.schema.training_schemas import EmbeddingDatasetInformation


class SimilarityMatrixEvaluation(BaseModel):
    """Represents the results from performing a similarity matrix eval on an embedding dataset.

    Evaluates similarity between n random datapoints using specified similarity function.
    """

    # Data Source
    test_dataset: EmbeddingDatasetInformation = Field(
        description="stitched or non-stitched testing dataset"
    )

    @model_validator(mode="after")
    def validate_dataset_is_test(self) -> "SimilarityMatrixPairwiseEvaluation":
        """Validate that this is a test dataset."""
        # TODO: Write this
        return self

    # Sampled Source Items
    record_ids: list[str] = Field(
        description="List of n record ids used for comparison",
        examples=[["00134-3434-2234", "02345-23157-3436"]],
    )

    # Process
    similarity_function: Literal["normalized_dot_product", "cosine_distance"] = Field(
        description="Name of similarity function used"
    )

    # Results
    similarity_matrix: list[list[float]] = Field(
        description="n x n matrix of similarity scores", ge=0, le=1
    )

    @computed_field
    @property
    def sample_size(self) -> int:
        """Get the number of samples based on record_ids length."""
        return len(self.record_ids)


class SimilarityMatrixPairwiseEvaluation(BaseModel):
    """Comparison metrics between different Similarity Matrix Evaluations."""

    # Similarity matrices being compared
    target_similarity_matrix: SimilarityMatrixEvaluation
    stitched_similarity_matrix: SimilarityMatrixEvaluation

    @model_validator(mode="after")
    def validate_matching_record_ids(self) -> "SimilarityMatrixPairwiseEvaluation":
        """Validate that both datasets have the same record IDs."""
        if (
            self.target_similarity_matrix.record_ids
            != self.stitched_similarity_matrix.record_ids
        ):
            raise ValueError(
                "Target and stitched datasets must have identical record IDs"
            )
        return self

    @computed_field
    @property
    def record_ids(self) -> list[str]:
        """Get the record IDs (same for both datasets after validation)."""
        return self.target_similarity_matrix.record_ids

    @computed_field
    @property
    def sample_size(self) -> int:
        """Get the sample size (same for both datasets after validation)."""
        return self.target_similarity_matrix.sample_size

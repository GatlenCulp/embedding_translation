"""Contains schema for our project."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.schema.training_schemas import (
    ExperimentConfig,
    EmbeddingMetadata,
    IngestionSettings,
    EvaluationSettings,
    TrainSettings,
    EmbeddingDatasetInformation,
    TrainStatus,
    StitchEvaluation,
    StitchEvaluationLog,
    StitchSummary
)

# StitchSummary is the main one I will be reading from

################################ VIZ (INTERFACE) SCHEMAS ##############################



class DatasetEvaluation(BaseModel):
    """Model to store info about how well EmbeddingDatasets perform.

    This is for either stitched datasets or source datasets.

    1. Sample n random datapoints
    2. Generate similarity matrix of these datapoints
    """
    test_dataset: EmbeddingDatasetInformation = Field(description="")
    top_k_metrics: dict[int, float]  # Maps k -> score for different k values
    label_accuracy: float  # How often desired document appears in results
    query_performance: dict[str, float]  # Maps query_id -> relevance score



class SimilarityMatrixEvaluation(BaseModel):
    """Represents the results from performing a similarity matrix eval on a dataset.

    Evaluates similarity between n random datapoints using specified similarity function.
    """

    dataset: EmbeddingDatasetInformation
    similarity_matrix: list[list[float]] = Field(description="n x n matrix of similarity scores")
    sample_size: int = Field(description="n samples chosen")
    record_ids: list[str] = Field(description="List of n record ids used for comparison")
    similarity_function: Literal["normalized_dot_product", "cosine_distance"] = Field(description="Name of similarity function used")


class SimilarityMatrixPairwaseEvaluation(BaseModel):
    """Comparison metrics between different Similarity Matrix Evaluations."""
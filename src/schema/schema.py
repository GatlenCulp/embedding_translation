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

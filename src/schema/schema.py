"""Contains schema for our project."""

from pydantic import BaseModel
from owler_fork.owlergpt.modern.schemas import (
    ExperimentConfig,
    EmbeddingMetadata,
    IngestionSettings,
    EvaluationSettings,
    TrainSettings,
    EmbeddingDatasetInformation,
    TrainStatus,
    StitchEvaluation,
    StitchEvaluationLog,
    StitchSummary,
)


################################ VIZ (INTERFACE) SCHEMAS ##############################


class DatasetEvaluation(BaseModel):
    """Model to store info about how well EmbeddingDatasets perform.

    This is for either stitched datasets or source datasets.
    """


class SimilarityMatrixDatasetEvaluation(BaseModel):
    """Represents the results from performing a similarity matrix eval on a dataset."""

    dataset_id: EmbeddingDatasetInformation


class DatasetComparisonEvaluation(BaseModel):
    """Comparison between different datasets."""

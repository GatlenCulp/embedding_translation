"""Contains schema for our project."""

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Define the path to the module
module_path = PROJECT_ROOT / "owler_fork" / "schemas.py"

# Load the module
spec = importlib.util.spec_from_file_location("schemas", module_path)
schemas = importlib.util.module_from_spec(spec)
sys.modules["schemas"] = schemas
spec.loader.exec_module(schemas)

# Import specific classes
ExperimentConfig = schemas.ExperimentConfig
EmbeddingMetadata = schemas.EmbeddingMetadata
IngestionSettings = schemas.IngestionSettings
EvaluationSettings = schemas.EvaluationSettings
TrainSettings = schemas.TrainSettings
EmbeddingDatasetInformation = schemas.EmbeddingDatasetInformation
TrainStatus = schemas.TrainStatus
DatasetEvaluation = schemas.DatasetEvaluation
DatasetComparison = schemas.DatasetComparison
StitchEvaluation = schemas.StitchEvaluation
StitchEvaluationLog = schemas.StitchEvaluationLog

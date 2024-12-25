"""Contains schema for our project. Unused shit"""

import importlib.util
import sys
from pathlib import Path

from pydantic import BaseModel


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
StitchEvaluation = schemas.StitchEvaluation
StitchEvaluationLog = schemas.StitchEvaluationLog


################################ VIZ (INTERFACE) SCHEMAS ################################


class DatasetEvaluation(BaseModel):
    """Model to store info about how well EmbeddingDatasets perform.

    This is for either stitched datasets or source datasets
    """

    # XXX - store information about how well datasets perform, i.e. whether they get the
    # desired document in the top k when we have labels


# XXX - gatlen
# 1. Kernel: idea is to sample n random datapoints and then just visualize a
#   similarity matrix (takes in chromaDB collection, entry type,
#   and what sort of similarity function to use). -> np table is output -> viz np table
# 2. CKA: calculate the kernel of the entire dataset for two matching datasets
#   (i.e. same text_dataset_name but different models) then
#   dot product it or something idk (it's defined somewhere - you find how different
#   the mega-tables are)
# 3. Rank: idk read owler_fork (basically it takes in a chromadb collection, and then
#   it calculates some wierd function of the top k results being compared
#    between one collection and another collection on the queries w.r.t.
#   the texts/documents; same for jaccard - basically read `owler_fork` and you will
#    iterate for different topk values and calculate some function of the IDs and
#   ranks of those topk results.
# 4. umap (do pca and tsne too maybe a couple times to be safe): just get all the
#   embeddings and umap this shit (then you need to color-code them
#    or partition them based on some sort of semantic classifier of your choice
# (i.e. you can manually classify documents using an LLM) and then
#    just color the UMAP)


class SimilarityMatrixDatasetEvaluation(BaseModel):
    """Represents the results from performing a similarity matrix eval on a dataset."""

    dataset_id: EmbeddingDatasetInformation


class DatasetComparisonEvaluation(BaseModel):
    """Comparison between different datasets."""

    # XXX - store information about how datasets store and query differently like cka, top k difference, etc...

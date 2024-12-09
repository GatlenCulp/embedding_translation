"""This is a definitive verison 1 for the schemas to how we store metadata about our experiments.

This is a WIP. You can copy it to have interoperability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Literal

from pydantic import BaseModel, computed_field, Field


################################ INGEST, EVAL, TRAIN SCHEMAS ################################


class EmbeddingMetadata(BaseModel):
    """ChromaDB embedding metadata.

    Every embedding in a ChromaDB collection has a metadata object that explains
    basically whether its a query or a document.
    """

    record_id: str
    chunk_id: str
    chunk_text: str
    record_text: str | None = None  # <---- like never used tbh
    record_type: Literal["query", "document"]
    # TODO(Adriano) not everything should be train by default
    record_split: Literal["train", "test"] = "train"
    tags: dict[str, str] | None = {}  # <---- should insert some meaning tags for umap cluster


class IngestionSettings(BaseModel):
    """Equivalent to owler's .env file.

    Used to define how to process the ingestion of the textual data into chromaDB
    """

    chunk_size: int = 256
    device: str | None = None  # does not matter
    distance_function: str | None = None  # does not matter
    normalize_embeddings: bool | None = None  # does not matter
    # TODO(Adriano) in the future we will want to try passing this through a model before
    chunk_preprocessing_mode: Literal["add_prefix"] = "add_prefix"
    query_preprocessing_mode: Literal["add_prefix"] = "add_prefix"
    chunk_prefix: str = (
        "passage: "  # Can be used to add prefix to text embeddings stored in vector store
    )
    query_prefix: str = (
        "query: "  # Can be used to add prefix to text embeddings used for semantic search
    )
    chunk_overlap: int = 25  # Determines, for a given chunk of text, how many tokens must overlap with adjacent chunks.
    dataloader_batch_size: int = 32
    dataloader_num_workers: int = 4


class EvaluationSettings(BaseModel):
    """Used at the evaluation stage."""

    chunk_size: int = 256
    embedding_dimension: int = 1024
    mean_center: bool = False  # Mean-center embedding vectors before calculating similarity
    k_nn_metric: str = "cosine"  # Metric to use for calculating nearest neighbors for exact query, see sklearn.metrics.pairwise.distance_metrics for allowed values
    k: int = 10  # The number of nearest neighbors to consider
    baseline: bool = False  # Compute baseline scores


class TrainSettings(BaseModel):
    """Train settings."""

    experiment_settings: ExperimentConfig  # <--- arch, src, dest, etc...

    # Training config
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10

    # Optimization
    optimizer: Literal["Adam"] = "Adam"
    optimizer_kwargs: dict[str, Any] | None = {}
    loss_fn: Literal["MSE"] = "MSE"
    loss_fn_kwargs: dict[str, Any] | None = {}


class EmbeddingDatasetInformation(BaseModel):
    """EmbeddingDataset info for Stitched or Non-Stitched Embeddings.

    The core unit of our analysis is an EMBEDDING DATASET which is associated primarily
    with a MODEL and an actual dataset (i.e. text). We do the following to embedding datasets:
    1. We create them from textual datasets
    2. We create them from other textual datasets
    3. We compare them using CKA, ranking metrics, etc...

    This object is SERIALIZEABLE and can be put into ChromaDB metadata AND it can alsobe stored in
    a JSON seperately.

    Every `EmbeddingDatasetInformation` always is created by:
    1. Ingesting
    2. Optionally stitch training; if not stitch-trained then it'll be None.
    (by having both of these infos we can trace exactly what went into this training run/experiment).
    """
    @computed_field
    @property
    def name(self) -> str:
        """Computes name for embedding dataset."""
        name = "EmbeddingDatasetInformation("
        f"embedding_model={self.embedding_model_type}/{self.embedding_model_name}, "
        f"text_dataset={self.text_dataset_source}/{self.text_dataset_name}, "
        ")"
        return name

    # Embedding model
    embedding_model_name: str
    embedding_model_type: Literal["openai", "huggingface"]
    embedding_dimension: int

    # Dataset + Index
    text_dataset_name: str  # <--- all from HF (but locally stored as .jsonl)
    text_dataset_source: Literal["huggingface"] = "huggingface"
    chromadb_collection_name: str  # <--- refers to ChromaDB OBJECT name

    # Ingestion parameters
    ingestion_settings: IngestionSettings
    stitch_train_settings: TrainSettings | None = None

    # Optional dataset + filepath
    dataset_filepath: str | None = None
    collections_filepath: str | None = None

    # Gatlen Proposed
    stitched: bool = False

    # Note: Should maybe also label which stitch model this was generated from if any.

    # XXX -adriano
    # 1. make dataset and dataloader for chromadb activations for layers if not available
    # 2. make good way of storing this shit in folders
    # 3. make sure to give a unique id to each embedding
    # 4. train
    # XXX :::::::::::::::::::::: files to be updated include (1) `train_linear_transforms_ds.py` and `schemas.py` here which might move.
    # XXX todo what is this
    # @staticmethod
    # def dataset_from_chromadb(self, chromadb_collection_name: str) -> Dataset:
    #     raise NotImplementedError("dataset_from_chromadb is an abstract method")
    # XXX not sure if I'm going to do this
    # class StitchPairDuringTraining(StitchPair):
    #     model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    #     stitch: nn.Linear
    #     optimizer: torch.optim.Optimizer
    #     loss_fn: nn.MSELoss

    #     def to_stitch_pair(self) -> StitchPair:
    #         """ `StitchPair` is meant to be serializeable but not `StitchPairDuringTraining`."""
    #         return StitchPair(
    #             model1=self.model1,
    #             model2=self.model2,
    #             mode=self.mode,
    #         )


class ExperimentConfig(BaseModel):
    """Configuration for a single stitch model training run."""

    @computed_field
    @property
    def name(self) -> str:
        """Generate a descriptive name for the stitch model training."""
        name = "Stitch("
        f"dataset={self.dataset_name} ({self.dataset_size} points), "
        f"architecture={self.architecture} (config={self.architecture_config})"
        f"source={self.source.embedding_model_name}, "
        f"target={self.target.embedding_model_name}, "
        ")"
        return name

    # Dataset config
    dataset_name: str  # e.g. "HotPotQA"
    dataset_split: Literal["train", "test"] = "train"
    dataset_train_test_split_frac: float = 0.8
    dataset_size: int

    # Source/Target embedding models
    source: EmbeddingDatasetInformation
    target: EmbeddingDatasetInformation

    # Architecture config
    architecture: Literal["affine"] = "affine"  # <--- which class
    architecture_config: dict[str, Any] | None = None  # <--- which kwargs for class


################################ DEBUGGING SCHEMAS ################################


# XXX subject to change
class TrainStatus(BaseModel):
    """Used to track whether training finished and where.

    TrainStatus is a file that should ONLY get updated at the
    START and END of training and is used primarily
    to track the status of the training. At the same time,
    you should, during training, be logging i.e. to a .log
    file or some other file, with any intermediate results you might care about.

    XXX maybe actually store per checkpoint or smth idk?
    """

    # Train stats:
    num_epochs: int = -1
    num_embeddings_per_epoch: int = -1
    num_embeddings_trained_on_total: int = (
        -1
    )  # <--- may include some extra if we crashed partway through
    train_info: dict[str, Any] | None = (
        None  # <--- store any sort of kwargs you might want to store
    )

    # Status
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    error: str | None = None

    # Linking to wandb
    wandb_run_project: str | None = None
    wandb_run_name: str | None = None

    # Timings
    time_start: datetime | None = None
    time_end: datetime | None = None
    time_taken_sec: float | None = None
    time_taken_min: float | None = None
    time_taken_hr: str | None = None

    # storage
    storage_info: dict[str, Any] | None = None


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


class StitchEvaluation(BaseModel):
    """When training a stitch. Single "snapshot" of how good it was at a point in time.

    Should mirror what we see in wandb.
    """

    # Timing info
    epoch_num: int
    num_embeddings_evaluated: int

    # Evaluation info
    stitching_mse: float
    stitching_mae: float
    stitching_additional_metrics: dict[str, Any] | None = {}  # <--- if you want more data use this
    evaluation_data_split: Literal["train", "test"] = "train"


class StitchEvaluationLog(BaseModel):
    """When training a stitch. Timeseries of snapshots."""

    evaluations: list[StitchEvaluation]

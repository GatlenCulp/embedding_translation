from __future__ import annotations

"""
This is a definitive verison 1 for the schemas to how we store metadata about our experiments. This is a WIP.
You can copy it to have interoperability.
"""


from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal
from datetime import datetime


class EmbeddingDatasetInformation(BaseModel):
    """
    The core unit of our analysis is an EMBEDDING DATASET which is associated primarily
    with a MODEL and an actual dataset (i.e. text). We do the following to embedding datasets:
    1. We create them from textual datasets
    2. We create them from other textual datasets
    3. We compare them using CKA, ranking metrics, etc...
    """
    model_name: str
    dataset_name: str
    collection_name: str
    model_dimension: int

    # NOTE: this is mostly copied from owlergpt environ variables
    distance_function: str = "cosine"

    dataset_filepath: Optional[str] = None
    collections_filepath: Optional[str] = None

class StitchPair(BaseModel):
    source: ModelInfo
    target: ModelInfo
    mode: Literal["affine"] = "affine" # TODO(Adriano) later we will support more shit here

class TrainStatus(BaseModel):
    """
    TrainStatus is a file that should ONLY get updated at the START and END of training and is used primarily
    to track the status of the training. At the same time, you should, during training, be logging i.e. to a .log
    file or some other file, with any intermediate results you might care about.
    """
    # Train stats:
    num_epochs: int = -1
    num_embeddings_per_epoch: int = -1
    num_embeddings_trained_on_total: int = -1 # <--- may include some extra if we crashed partway through
    train_info: Optional[Dict[str, Any]] = None # <--- store any sort of kwargs you might want to store
    
    # Status
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    error: Optional[str] = None

    # Linking to wandb
    wandb_run_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Timings
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    time_taken_sec: Optional[float] = None
    time_taken_min: Optional[float] = None
    time_taken_hr: Optional[str] = None

    # storage
    storage_info: Optional[Dict[str, Any]] = None

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
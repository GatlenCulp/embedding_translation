from __future__ import annotations
from typing import List, Tuple, Literal, Optional, Dict, Any
import yaml
import os
import click
import wandb
import chromadb
import numpy as np
import pandas as pd
import safetensors
import torch
import torch.nn as nn
import itertools
import pydantic
from pydantic import BaseModel
from flask import current_app
from tqdm import tqdm
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from owlergpt.utils.cli_helpers import get_selected_folder, get_chroma_collections
from owlergpt.modern.collection_utils import parse_collection_name, model2model_dimension

# XXX use the defintive schema definitions (Etc...)
class ModelInfo(BaseModel):
    model_name: str
    collection_name: str
    model_dimension: int

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

class LinearTransformTrainer:
    def __init__(self, linear: nn.Linear, pair: StitchPair, save_folder: Path):
        self.linear = linear
        self.pair = pair
        self.train_status = TrainStatus()
        self.save_folder = save_folder

        self.train_status_file = self.save_folder / "train_status.json"
        self.log_duplication_file = self.save_folder / "log_duplication.json"
        self.linear_safetensors_file = self.save_folder / "linear_transform.safetensors"
        self.pair_info_file = self.save_folder / "pair_info.json"
        self.checkpoints_dir = self.save_folder / "checkpoints"
    
    # XXX TODOs are the following:
    # 1. be able to create the folders and write train status at the end and update it at the end
    # 4. Be able to load chromas

    def write_train_status(self, train_status: TrainStatus):
        with open(self.save_folder / "train_status.json", "w") as f:
            f.write(train_status.model_dump_json(indent=4))

    def train(
            self,
            num_epochs: int,
            train_info: Optional[Dict[str, Any]] = None,
    ):
        pass # XXX

@current_app.cli.command("train_ds")
def train_linear_transforms_dataset() -> None:
    """
    This simple script will train AFFINE linear transformations between each ordered pair of
    embedded models' datasets. We can see in the $DATASET_FOLDER_PATH the datasets that will be used
    (to get their names) and then use this to get the relevant collections from Chroma DB. We then
    train for all non-equal ordered pairs (so, bidirectionally: n(n-1) pairs if there are n models)
    on that dataset, linear layers between each of the two. This is logged to `wandb` and information
    about the specific model trained, etc... is stored to a folder in $TRAIN_FOLDERS_PATH. If any of
    these environment variables are not set (including $WANDB_NAME and $WANDB_PROJECT) then the
    script will throw an error. Unfortunately, it is meant, right now, to run only on specific datasets.

    The output is that we populate the $TRAIN_FOLDERS_PATH with a bunch of subfolders:
        $TRAIN_FOLDERS_PATH/<uuid>/
            pair_info.json               # <--- the thing to use to know what this is translating between
            linear_transform.safetensors # <--- the thing to test
            train_status.json            # <--- the thing to use to be fault tolerant etc...
            ... (maybe some log files, checkpoints, etc...)
    """
    environ = os.environ
    # default_chunk_size = int(environ.get("VECTOR_SEARCH_SENTENCE_DEFAULT_CHUNK_SIZE", 100))
    # target_dimension = int(environ.get("EMBEDDING_DIMENSION"))  # Target dimension for the embeddings
    # chunk_size = environ.get("CHUNK_SIZE")
    # center = bool(int(environ.get("MEAN_CENTER")))
    # batch_size = int(environ.get("BATCH_SIZE"))
    device = torch.device(environ["VECTOR_SEARCH_SENTENCE_TRANSFORMER_DEVICE"])
    if device == "cpu":
        raise ValueError("CPU is not supported for training (because it'll take so long ur trolling dummy)")
    # nn_metric = environ.get("K_NN_METRIC")
    # baseline = bool(int(environ.get("BASELINE")))
    k = int(environ.get("K"))
    if k < 3:
        raise ValueError("Number of retrieved results must be at least 3")

    print("Initializing Chroma + Fetching valid datasets/models...")
    chroma_client = chromadb.PersistentClient(
        path=environ["VECTOR_SEARCH_PATH"],
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    selected_folder = get_selected_folder(environ)
    collections: List[str] = get_chroma_collections(chroma_client, selected_folder)
    model_names: List[str] = [parse_collection_name(collection)[1] for collection in collections]
    model_dims: List[int] = [model2model_dimension(model_name) for model_name in model_names]
    model_infos: List[ModelInfo] = [
        ModelInfo(
            model_name=model_name,
            collection_name=collection,
            model_dimension=model_dim,
        )
        for model_name, collection, model_dim in zip(model_names, collections, model_dims)
    ]
    print("\n".join(mf.model_dump_json(indent=4) for mf in model_infos))

    train_pairs: List[StitchPair] = list(itertools.product(model_infos, model_infos))
    train_pairs = [pair for pair in train_pairs if pair[0].model_name != pair[1].model_name]
    assert len(train_pairs) == len(model_infos) * (len(model_infos) - 1)
    print(f"Will train on {len(train_pairs)} pairs")

    # NOTE: we will use wandb_name as a prefix
    if "WANDB_PROJECT" not in environ:
        raise ValueError("WANDB_PROJECT must be set")
    if "WANDB_NAME" in environ: # <--- we create the names
        raise ValueError("WANDB_NAME must be set")
    if "TRAIN_FOLDER_PATH" not in environ:
        raise ValueError("TRAIN_FOLDER_PATH must be set")

    train_folder_path = Path(environ["TRAIN_FOLDER_PATH"])
    train_folder_path.mkdir(parents=True, exist_ok=True)

    # Create subfolders and save data for each pair
    # TODO(Adriano) consider looking for duplicates?
    for pair in tqdm(train_pairs, desc="Creating/preparing subfolders for each pair"):
        linear = nn.Linear(pair[0].model_dimension, pair[1].model_dimension, bias=True).to(device)
        # Validate model names don't contain slashes
        assert isinstance(pair[0], ModelInfo)
        assert isinstance(pair[1], ModelInfo)
        assert "/" not in pair[0].model_name, f"Model name should not contain '/': {pair[0].model_name}"
        assert "/" not in pair[1].model_name, f"Model name should not contain '/': {pair[1].model_name}"

        # 0. Create subfolder with UUID
        subfolder = train_folder_path / str(uuid4())
        subfolder.mkdir(parents=True, exist_ok=True)

        # 1. Save pair info as JSON using pydantic model
        pair_info_path = subfolder / "pair_info.json"
        with open(pair_info_path, "w") as f:
            f.write(StitchPair(source=pair[0], target=pair[1]).model_dump_json(indent=4))

        # 1.5 Setup wandb
        wandb_run_project = environ["WANDB_PROJECT"] # XXX
        wandb_run_name = str(uuid4()) # XXX

        # 2. Save train info 1
        train_info_path = subfolder / "train_info.json"
        with open(train_info_path, "w") as f:
            f.write(TrainStatus(
                num_epochs=1,
                num_embeddings_per_epoch=100, # XXX
                num_embeddings_trained_on_total=100, # XXX
                train_info={}, # XXX
                time_start=datetime.now(),
                status="running",
                storage_info={}, # XXX
            ).model_dump_json(indent=4)) # XXX

    for pair in tqdm(train_pairs, desc="Training..."):
        trainer = LinearTransformTrainer(linear, pair, subfolder)
        wandb.init(project=wandb_run_project, name=wandb_run_name)

        # 3. Train the linear transform
        trainer.train(num_epochs=1)

        # 4. Create dummy tensor with appropriate dimensions and save as safetensors
        linear_safetensors_path = subfolder / "linear_transform.safetensors"
        safetensors.torch.save_file(linear.state_dict(), linear_safetensors_path)

        # 5 Update train info
        pass # XXX

        # 6. Close wandb - we use a different wandb run for each pair
        wandb.finish()


    raise NotImplementedError("TODO(Adriano): implement this") # XXX
    

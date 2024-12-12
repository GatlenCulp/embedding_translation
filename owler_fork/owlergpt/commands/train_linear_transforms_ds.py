from __future__ import annotations

"""
Shitty train script. This is NOT meant to be used later when it is replaced by the actual schemas. Does not actually log everything. Is
really jank ngl. 

[WARNING: DEPRECATED] (try to use the `train_on_safetensors_dbs.ipynb`)
"""
from typing import List, Tuple, Literal, Optional, Dict, Any
import yaml
import os
import click
from torch.utils.data import DataLoader, Dataset
import wandb
import chromadb
import numpy as np
import pandas as pd
import safetensors
import torch
import torch.nn as nn
import itertools
import time
import pydantic
import json
from pydantic import BaseModel
from flask import current_app
from tqdm import tqdm
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from owlergpt.utils.cli_helpers import get_selected_folder, get_chroma_collections
from owlergpt.modern.collection_utils import parse_collection_name, model2model_dimension

class ModelInfo(BaseModel):
    model_name: str
    collection_name: str
    model_dimension: int


class StitchPair(BaseModel):
    source: ModelInfo
    target: ModelInfo
    dataset: str
    mode: Literal["affine"] = "affine"  # TODO(Adriano) later we will support more shit here

    def save_linear_transform(self, linear: nn.Linear, save_path: Path) -> None:
        linear_path = save_path / "linear_transform.safetensors"
        stitch_info_path = save_path / "stitch_info.json"
        assert not linear_path.exists(), f"Linear transform already exists at {linear_path}"
        assert not stitch_info_path.exists(), f"Stitch info already exists at {stitch_info_path}"
        safetensors.torch.save_file(linear.state_dict(), linear_path)
        stitch_info_path.write_text(self.model_dump_json())


# TODO(Adriano) actually use the train_status file plz
# class TrainStatus(BaseModel):
#     """
#     TrainStatus is a file that should ONLY get updated at the START and END of training and is used primarily
#     to track the status of the training. At the same time, you should, during training, be logging i.e. to a .log
#     file or some other file, with any intermediate results you might care about.
#     """
#     # Train stats:
#     num_epochs: int = -1
#     num_embeddings_per_epoch: int = -1
#     num_embeddings_trained_on_total: int = -1 # <--- may include some extra if we crashed partway through
#     train_info: Optional[Dict[str, Any]] = None # <--- store any sort of kwargs you might want to store

#     # Status
#     status: Literal["pending", "running", "completed", "failed"] = "pending"
#     error: Optional[str] = None

#     # Linking to wandb
#     wandb_run_project: Optional[str] = None
#     wandb_run_name: Optional[str] = None

#     # Timings
#     time_start: Optional[datetime] = None
#     time_end: Optional[datetime] = None
#     time_taken_sec: Optional[float] = None
#     time_taken_min: Optional[float] = None
#     time_taken_hr: Optional[str] = None

#     # storage
#     storage_info: Optional[Dict[str, Any]] = None


def get_train_test_set_ids(
    chroma_client: chromadb.PersistentClient,
    collections: List[str],
    record_types: List[str],
    train_test_split: float = 0.8,
) -> Tuple[List[str], List[str]]:
    # Get all record IDs from one of the collections
    print("Getting the initial record ids...")
    collection1 = chroma_client.get_collection(collections[0])
    print("> Filtering by record type...")
    records = collection1.get(
        # {"record_id": ids[i], "record_text": text_chunks[i], "record_type": record_type} = metadata per record
        where={"record_type": {"$in": record_types}}
    )
    # Get record_ids from metadata since that's where they are stored
    print("> Getting record ids from metadata...")
    _ = [record["record_id"] for record in records["metadatas"]]
    print("> Setting record ids...")
    collection_record_ids = set(_)
    assert len(collection_record_ids) == len(_)  # all shud be unique or we die

    for collection_name in tqdm(collections, desc="Verifying all collections have the same record IDs..."):
        collection = chroma_client.get_collection(collection_name)
        records = collection.get(where={"record_type": {"$in": record_types}})
        record_ids = set([record["record_id"] for record in records["metadatas"]])
        assert record_ids == collection_record_ids  # if not all are same record id i cry a river

    # random split ftw
    print("Getting the splits (I'm no acrobat doe)")
    print("> Shuffling...")
    collection_record_ids = list(collection_record_ids)
    rng = np.random.default_rng(42)
    rng.shuffle(collection_record_ids)
    print("> Splitting...")
    num_train = int(train_test_split * len(collection_record_ids))
    train_ids = collection_record_ids[:num_train]
    test_ids = collection_record_ids[num_train:]
    assert 0 < len(train_ids) < len(collection_record_ids)
    assert 0 < len(test_ids) < len(collection_record_ids)
    return train_ids, test_ids


class StitchingDataset(Dataset):
    """Inefficiently and simply implemeneted stitching dataset ay lmao."""

    def __init__(
        self,
        source_collection: chromadb.Collection,
        target_collection: chromadb.Collection,
        # {"record_id": ids[i], "record_text": text_chunks[i], "record_type": record_type} = metadata per record
        record_ids: List[str],
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """Dataset for training linear transformations between embeddings.

        Args:
            source_collection: ChromaDB collection containing source embeddings
            target_collection: ChromaDB collection containing target embeddings
            record_ids: List of record IDs to use from both collections
        """
        self.source_collection = source_collection
        self.target_collection = target_collection
        self.record_ids = record_ids

        # Verify collections have the specified IDs
        # {"record_id": ids[i], "record_text": text_chunks[i], "record_type": record_type} = metadata per record
        # find where metadata "record_id" matches the record ids
        source_records = source_collection.get(where={"record_id": {"$in": record_ids}})  # fmt: skip
        target_records = target_collection.get(where={"record_id": {"$in": record_ids}})  # fmt: skip

        if len(source_records["ids"]) != len(record_ids):
            raise ValueError("Source collection missing some record IDs")
        if len(target_records["ids"]) != len(record_ids):
            raise ValueError("Target collection missing some record IDs")

        # Convert embeddings to tensors
        # TODO(Adriano) is this acceptable? seems like from back of the envelope its not even a gig:
        # `2000 * 10000 * 8 / 1000 / 1000` (I guess we regularly load billion parameter models, this should be OK for small models!)
        self.source_embeddings = torch.tensor(source_records["embeddings"], dtype=torch.float32).to(
            device
        )
        self.target_embeddings = torch.tensor(target_records["embeddings"], dtype=torch.float32).to(
            device
        )

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, idx):
        return self.source_embeddings[idx], self.target_embeddings[idx]


def create_train_test_datasets(
    source_collection: chromadb.Collection,
    target_collection: chromadb.Collection,
    train_ids: List[str],
    test_ids: List[str],
) -> Tuple[StitchingDataset, StitchingDataset]:
    """Creates train and test datasets from two collections using provided ID splits.

    Args:
        source_collection: ChromaDB collection containing source embeddings
        target_collection: ChromaDB collection containing target embeddings
        train_ids: List of record IDs for training set
        test_ids: List of record IDs for test set

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_dataset = StitchingDataset(source_collection, target_collection, train_ids)
    test_dataset = StitchingDataset(source_collection, target_collection, test_ids)
    return train_dataset, test_dataset

# TODO(Adriano) get more dank by training multiple of these dummies at once, check this out
# 10 vectors = 1 vector 10 times long cat cat cat cat cat cat cat cat cat cat
# then you do mse on the entire damn thing because thats the same as mse seperately
# then you get 10x boost on H100 cuz its a husky Bpu (based process unit)
# embrace husky bpu
class LinearTransformTrainer:
    def __init__(
        self,
        save_path: Path,
        linear: nn.Linear,
        source_collection: str,
        target_collection: str,
        chroma_client: chromadb.PersistentClient,
        device: torch.device | str,
        train_ids: List[str],
        test_ids: List[str],
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        save_every_n_epochs: int = 10,
    ):
        self.linear = linear
        self.source_collection = source_collection
        self.target_collection = target_collection
        self.chroma_client = chroma_client
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=self.learning_rate)
        self.device = device
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.save_every_n_epochs = save_every_n_epochs
        self.save_path = save_path
        self.checkpoint_path = save_path / "checkpoints"
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.logfile = save_path / "log.jsonl"

    def train(self):
        train_dataset, test_dataset = create_train_test_datasets(
            self.source_collection,
            self.target_collection,
            self.train_ids,
            self.test_ids,
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        mse_loss = nn.MSELoss()
        for epoch in range(self.num_epochs):
            # Training
            self.linear.train()
            train_mse = 0.0
            train_mae = 0.0
            num_train_batches = 0

            for source_emb, target_emb in train_loader:
                source_emb = source_emb.to(self.device)
                target_emb = target_emb.to(self.device)

                self.optimizer.zero_grad()
                output = self.linear(source_emb)

                loss = mse_loss(output, target_emb)
                loss.backward()
                self.optimizer.step()

                train_mse += loss.detach().item()
                train_mae += (output.detach() - target_emb.detach()).abs().mean().item()
                num_train_batches += 1

            avg_train_mse = train_mse / num_train_batches
            avg_train_mae = train_mae / num_train_batches

            # Evaluation
            self.linear.eval()
            test_mse = 0.0
            test_mae = 0.0
            num_test_batches = 0

            with torch.no_grad():
                for source_emb, target_emb in test_loader:
                    source_emb = source_emb.to(self.device)
                    target_emb = target_emb.to(self.device)

                    output = self.linear(source_emb)

                    test_mse += mse_loss(output, target_emb).item()
                    test_mae += (output.detach() - target_emb.detach()).abs().mean().item()
                    num_test_batches += 1

            avg_test_mse = test_mse / num_test_batches
            avg_test_mae = test_mae / num_test_batches

            # Log metrics
            log_entry = {
                "epoch": epoch,
                "train_mse": avg_train_mse,
                "train_mae": avg_train_mae,
                "test_mse": avg_test_mse,
                "test_mae": avg_test_mae,
            }
            wandb.log(log_entry)
            with open(self.logfile, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if epoch % self.save_every_n_epochs == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch: int):
        checkpoint_path = self.checkpoint_path / f"checkpoint_{epoch}.safetensors"
        safetensors.torch.save_file(self.linear.state_dict(), checkpoint_path)


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
        raise ValueError(
            "CPU is not supported for training (because it'll take so long ur trolling dummy)"
        )
    print("Using device:", device, "(hella epic - icelandic water™ brings you viking power)")
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
    # get rid of ultra_debug_small collections
    collections_who_most_die = [collection for collection in chroma_client.list_collections() if "ultra_debug_small" in collection.name]
    print(f"Deleting {len(collections_who_most_die)} collections")
    print("  " + "\n  ".join(collection.name for collection in collections_who_most_die))
    click.confirm("Continue?", abort=True)
    for collection in tqdm(list(collections_who_most_die), desc="Deleting collections"):
        chroma_client.delete_collection(collection.name)
    
    collections: List[str] = get_chroma_collections(chroma_client, selected_folder)
    SHRINK_COLLECTIONS_FOR_TESTING = True # comment/uncomment
    if SHRINK_COLLECTIONS_FOR_TESTING:
        print("TIME TO SHRINK ---- we musT DESCEND into the QUANTUM REALM....;;;::::::>>>>>>>>.......````")
        # small collection => good testing
        print("------- [quantum enabled] getting collection objs")
        collection_objs = [chroma_client.get_collection(collection) for collection in collections]
        limit = 1
        print(f"------- [quantum enabled] getting small subset data (limit={limit})")
        small_subset_data = [collection_obj.get(limit=limit) for collection_obj in tqdm(collection_objs, desc="Getting small subset data")]
        print("------- [quantum enabled] creating new collections")
        new_collections = [chromadb.Collection(name=f"ultra_debug_small_{collection.name}") for collection in collections]
        print("------- [quantum enabled] adding small subset data to new collections")
        for new_collection, subset_data in tqdm(list(zip(new_collections, small_subset_data)), desc="Adding small subset data to new collections"):
            new_collection.add(
                embeddings=subset_data['embeddings'],
                documents=subset_data['documents'],
                metadatas=subset_data['metadatas'],
                ids=subset_data['ids']
            )
        print("[quantum enabled] sans dat")
        assert len(collections) == len(new_collections)
        assert all(nc.name == f"small_{c.name}" for nc, c in zip(new_collections, collections))
        print("[quantum enabled] replac ->>>> our journey is complete [returning <<>> to MACROSCOPIC world]........ | E X  P   A    N     D || |  |   |    |     |")
        collections = new_collections # replace ftw
    print(f"Found {len(collections)} collections")
    # We are only training on the documents right now
    print("Fetching train/test set IDs... and making sure this shit aint too sussy backa (hella viz in there too nw, tqdm, and even print statements)")
    train_set_ids, test_set_ids = get_train_test_set_ids(
        chroma_client, collections, record_types=["document"]
    )
    print(f"Train set size: {len(train_set_ids)}")
    print(f"Test set size: {len(test_set_ids)}")

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
    click.confirm("Continue?", abort=True)

    train_pairs: List[StitchPair] = list(itertools.product(model_infos, model_infos))
    train_pairs = [pair for pair in train_pairs if pair[0].model_name != pair[1].model_name]
    assert len(train_pairs) == len(model_infos) * (len(model_infos) - 1)
    print(f"Will train on {len(train_pairs)} pairs")
    click.confirm("Continue?", abort=True)
    # NOTE: we will use wandb_name as a prefix
    if "WANDB_PROJECT" not in environ:
        raise ValueError("WANDB_PROJECT must be set")
    if "WANDB_NAME" in environ:  # <--- we create the names
        raise ValueError("WANDB_NAME must be set")
    if "TRAIN_FOLDER_PATH" not in environ:
        raise ValueError("TRAIN_FOLDER_PATH must be set")

    train_folder_path = Path(environ["TRAIN_FOLDER_PATH"])
    train_folder_path.mkdir(parents=True, exist_ok=True)

    # Create subfolders and save data for each pair
    # TODO(Adriano) consider looking for duplicates?
    print("<><><><><><>< IT IS TIME ><><>< ... I am braking my chains... 0%... 5%.......... 29%..... 420%... UNLEASH THE KRAKEN~~~~ ! DOOM SHALL BEFALL THIS WORLD <><><><><><>< ~~~<~><><><~>~<>~<><~>~<~> UUAOUAOUOUOGHH~<")
    for pair in tqdm(train_pairs, desc="Training each pair of models' stitching..."):
        start_time = time.time()
        linear = nn.Linear(pair[0].model_dimension, pair[1].model_dimension, bias=True).to(device)
        # Validate model names don't contain slashes
        assert isinstance(pair[0], ModelInfo)
        assert isinstance(pair[1], ModelInfo)
        assert (
            "/" not in pair[0].model_name
        ), f"Model name should not contain '/': {pair[0].model_name}"
        assert (
            "/" not in pair[1].model_name
        ), f"Model name should not contain '/': {pair[1].model_name}"

        # 0. Create subfolder with UUID
        subfolder = train_folder_path / str(uuid4())
        subfolder.mkdir(parents=True, exist_ok=True)

        # 1. Save pair info as JSON using pydantic model
        pair_info_path = subfolder / "pair_info.json"
        with open(pair_info_path, "w") as f:
            # chunk size alwasys 256
            # all dat shit is default AF
            f.write(StitchPair(source=pair[0], target=pair[1]).model_dump_json(indent=4))

        # 1.5 Setup wandb
        wandb_run_project = environ["WANDB_PROJECT"]  # ay lmao
        wandb_run_name = pair[0].model_name + "_" + pair[1].model_name + "_" + str(uuid4())
        wandb.init(project=wandb_run_project, name=wandb_run_name)

        # 2. Train the linear transform
        trainer = LinearTransformTrainer(
            save_path=subfolder,
            linear=linear,
            source_collection=pair[0].collection_name,
            target_collection=pair[1].collection_name,
            chroma_client=chroma_client,
            device=device,
            train_ids=train_set_ids,
            test_ids=test_set_ids,
            num_epochs=50,
            batch_size=32,
            learning_rate=0.001,
            save_every_n_epochs=5,
        )
        trainer.train()

        # 4. Create dummy tensor with appropriate dimensions and save as safetensors
        linear_safetensors_path = subfolder / "linear_transform.safetensors"
        safetensors.torch.save_file(linear.state_dict(), linear_safetensors_path)

        # 5 Update train info
        with open(subfolder / "train_status.json", "w") as f:
            f.write(
                json.dumps(
                    {
                        "status": "completed",
                        "time_taken_sec": time.time() - start_time,
                    },
                    indent=4,
                )
            )

        # 6. Close wandb - we use a different wandb run for each pair
        wandb.finish()

from __future__ import annotations
import click
from chromadb.api.models.Collection import Collection
from typing import List, Tuple, Literal, Optional, Dict, Any
import os
import click
from typing import List
from chromadb.config import DEFAULT_TENANT
from flask import current_app
from owlergpt.utils import choose_dataset_folders
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
from chromadb.config import DEFAULT_TENANT
from flask import current_app
from owlergpt.utils.cli_helpers import get_selected_folder, get_chroma_collections
from owlergpt.modern.schemas import EmbeddingDatasetInformation, EmbeddingMetadata
from tqdm import tqdm
from pathlib import Path
from uuid import uuid4
from datetime import datetime


def update_collection_metadata(collection: Collection, metadata_key: str, new_value: any):
    results = collection.get()
    for idx, metadata in enumerate(results['metadatas']):
        metadata[metadata_key] = new_value
        collection.update(
            ids=results['ids'][idx],
            metadatas=metadata
        ) # XXX

@current_app.cli.command("ingest_ds")
def split_train_test_ds() -> None:
    split_frac = click.prompt("Enter the fraction of the dataset to be used for training", type=float, default=0.8)
    if split_frac < 0 or split_frac > 1:
        raise ValueError("Fraction must be between 0 and 1")
    print(f"Splitting dataset with fraction {split_frac} for training and {1 - split_frac} for testing")


    assert os.environ.get("VECTOR_SEARCH_PATH") is not None
    assert os.environ.get("DATASET_FOLDER_PATH") is not None

    selected_folder = get_selected_folder(os.environ)
    chroma_client = chromadb.PersistentClient(
        path=os.environ["VECTOR_SEARCH_PATH"],
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collections = get_chroma_collections(chroma_client, selected_folder)
    print(collections) # XXX DEBUG

    # record_id: str
    # record_text: str 
    # record_type: Literal["query", "document"]
    # # TODO(Adriano) not everything should be train by default
    # record_split: Literal["train", "test"] = "train"
    # tags: Optional[Dict[str, str]] = {} # <---- should insert some meaning tags for umap cluster
    

    for collection in tqdm(collections, desc="Updating collection metadata"):
        print(collection) # XXX DEBUG
        update_collection_metadata(collection, "split", "train")

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
import tempfile
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
from owlergpt.modern.schemas import EmbeddingDatasetInformation, EmbeddingMetadata, IngestionSettings
from owlergpt.modern.collection_utils import OPENAI_MODELS, parse_collection_name, model2model_dimension, MODEL_NAMES
from owlergpt.utils import JSONDataset, collate_fn
from tqdm import tqdm
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter, TokenTextSplitter


@current_app.cli.command("metadata_ds")
def fix_metadata_ds() -> None:
    """
    This script will 
    """
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
    collection_names: List[str] = get_chroma_collections(chroma_client, selected_folder, enforce_valid=True)
    print("="*40 + " Will modify the following collections:" + "="*40)
    print('\n'.join(collection_names))
    print("="*80)
    # record_id: str
    # record_text: str 
    # record_type: Literal["query", "document"]
    # # TODO(Adriano) not everything should be train by default
    # record_split: Literal["train", "test"] = "train"
    # tags: Optional[Dict[str, str]] = {} # <---- should insert some meaning tags for umap cluster
    
    raise NotImplementedError("Not implemented") # XXX
    # coordinator = EmbeddingMetadataPopulatorCoordinator(
    #     selected_folder,
    #     collection_names,
    #     MetadataPopulatorArgs(split_frac, overwrite_metadata=True),
    # )
    # coordinator.main()


@current_app.cli.command("test_metadata_ds")
def test_fix_metadata_ds():
    """
    Test command you should run to make sure that this works OK. Does:
    1. A test on a fake dataset
    2. A dry-run on the real dataset without actually changing anything (this might not always work as intended ngl)
    """
    print("="*40 + " Testing me" + "="*40)
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("CUDA must be available and CUDA_VISIBLE_DEVICES must be set")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA is not available, using CPU")

    # default parameters
    selected_folder = "test_dataset"
    tokens_per_chunk = 256
    chunk_overlap = 25

    # Filter for small models, cap at 3
    small_model_names = [m for m in MODEL_NAMES if "small" in m.lower()][:3]
    assert all("/" in m for m in small_model_names), f"Expected all small models to be HF models, got {small_model_names}" # fmt: skip
    assert len(small_model_names) == 3, f"Expected 3 small models, got {len(small_model_names)}" # should be multiple
    if len(small_model_names) == 0:
        raise ValueError("No small models found in OPENAI_MODELS")
    models = [SentenceTransformer(model_name, device=device) for model_name in small_model_names]
    model_names = [s_model_name.split("/")[-1] for s_model_name in small_model_names] # get the model names for saving
    text_splitters = [
        SentenceTransformersTokenTextSplitter(
            model_name=s_model_name,
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk
        )
        for s_model_name in small_model_names
    ]
    print(small_model_names)
    print(models)
    print(model_names)
    print(text_splitters)

    raise NotImplementedError("Not implemented") # XXX

    # # Create test sentences + ids
    # test_sentences = [
    #     "The quick brown fox jumps over the lazy dog.",
    #     "A journey of a thousand miles begins with a single step.",
    #     "All that glitters is not gold.",
    #     "Actions speak louder than words.",
    #     "Beauty is in the eye of the beholder.",
    #     "Every cloud has a silver lining.",
    #     "Fortune favors the bold.",
    #     "Knowledge is power.",
    #     "Practice makes perfect.",
    #     "Time heals all wounds.",
    #     # add one long entry here so that we can pass the 256 limit
    #     "donkey is happy, " * 400, # surely at least 2 chunks at least with toks
    # ]
    # record_ids = [f"test_record_{i}" for i in range(len(test_sentences))]


    # # Initialize ChromaDB client in temporary directory and do everything in there to avoid any mistakes
    # # TODO(Adriano) we should have some sort of tokenization abstraction to bring this into the ingest_ds.py
    # #   sort of functionality
    # with tempfile.TemporaryDirectory() as temp_dir:
    #     chroma_client = chromadb.PersistentClient(
    #         path=temp_dir,
    #         settings=chromadb.Settings(anonymized_telemetry=False)
    #     )

    #     collections = []
    #     # Create collections and populate with embeddings
    #     for i, model in tqdm(enumerate(models), desc="Creating collections"):
    #         # Create collection
    #         # NOTE this has to be parseable
    #         # `{selected_folder}_{transformer_model}_CharacterSplitting_{tokens_per_chunk}`
    #         collection_name = f"{selected_folder}_{model_names[i]}_CharacterSplitting_{tokens_per_chunk}"
    #         collection = chroma_client.create_collection(
    #             name=collection_name,
    #             metadata={"hnsw:space": "cosine"}  # Using cosine as default distance function
    #         )
            
    #     # Generate embeddings
    #     embeddings = model.encode(test_sentences, convert_to_tensor=False)
        
    #     # Create metadata for each embedding
    #     metadatas = [
    #         {
    #             "record_id": record_ids[i],
    #             "record_text": test_sentences[i],
    #             "record_type": "document"
    #         }
    #         for i in range(len(test_sentences))
    #     ]
        
    #     # Add to collection
    #     collection.add(
    #         embeddings=embeddings.tolist(),
    #         documents=test_sentences,
    #         metadatas=metadatas,
    #         ids=record_ids
    #     )
        
    #     collections.append(collection)

    # return collections, temp_dir
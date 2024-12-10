from __future__ import annotations
import sys
from pathlib import Path
_ = Path(__file__).parent.parent.parent
print("=====================> Adding to sys.path", _)
sys.path.append(_.as_posix()) # VIEW owlergpt
print("Current sys.path:", sys.path)
import time
from typing import List, Tuple, Callable
import torch
import click
import os
import gc
from flask import current_app
import torch.nn as nn
import safetensors
import json
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from sentence_transformers import SentenceTransformer
from owlergpt.utils.json_loader import JSONDataset, collate_fn
from owlergpt.modern.collection_utils import MODEL_NAMES, OPENAI_MODELS
import torch.multiprocessing as mp
from owlergpt.modern.collection_utils import model2model_dimension

# NOTE copied from chunk_dataset.py
class Chunk(BaseModel):
    id: str = Field(alias="id") # unique id for the chunk (can combine doc id with index within that chunk)
    doc_id: str = Field(alias="doc_id")
    index_in_doc: int = Field(alias="index_in_doc")
    text: str = Field(alias="text")

mp.set_start_method('spawn', force=True)
NON_OPENAI_MODELS = [m for m in MODEL_NAMES if not any (oai in m for oai in OPENAI_MODELS)]

DATASETS = [
    # (numbers are counts for documents, there may be some longer documents -> slightly more chunks)
    "arguana", # 10K
    "fiqa", # 50K -> 20K
    "scidocs", # 25K -> 20K
    "nfcorpus", # 5K
    "hotpotqa", # 100K -> 20K
    "trec-covid", # too much -> 20K
]

class SingletonInjestor:
    """
    Ingest embeddings by computing them. Do this serially. TODO do batch later.
    """
    def __init__(
            self,
            output_dir: Path,
            device: str,
            parallelize: bool,
            dataset_chosen: str,
    ):
        self.dataset_chosen = dataset_chosen
        self.parallelize = parallelize
        self.output_dir = output_dir
        # self.chunk_size = 256
        # self.chunk_overlap = 25
        self.chunk_size = int(os.environ.get("CHUNK_SIZE", None))
        self.chunk_overlap = int(os.environ.get("VECTOR_SEARCH_TEXT_SPLITTER_CHUNK_OVERLAP", None))
        # self.normalize_embeddings = False
        self.normalize_embeddings = bool(os.environ.get("VECTOR_SEARCH_NORMALIZE_EMBEDDINGS", None))
        self.batch_size = int(os.environ.get("BATCH_SIZE", None))
        self.chunk_prefix = os.environ.get("VECTOR_SEARCH_CHUNK_PREFIX", None)
        self.query_prefix = os.environ.get("VECTOR_SEARCH_QUERY_PREFIX", None)
        _NON_OPENAI_MODELS = [m for m in NON_OPENAI_MODELS if m != "Salesforce/SFR-Embedding-Mistral"]
        print(f"CAN USE {len(_NON_OPENAI_MODELS)} NON OPENAI MODELS")
        print("  " + "\n  ".join(_NON_OPENAI_MODELS))
        chunks_path = Path(__file__).parent.parent.parent / "chunks" / self.dataset_chosen
        self.datasets_path = chunks_path # lol
        self.transformer_names: List[List[str]] = [[m for m in _NON_OPENAI_MODELS]]
        print("="*50)
        print('\n'.join(f"({i}) {m}" for i, m in enumerate(_NON_OPENAI_MODELS)))
        print("="*50)
        self.device = device
        print("SELF DEVICE IS ", self.device)
    
    def h__save_embedding_results(
        self,
        embedding: torch.Tensor,
        ids: List[str],
        documents: List[str],
        chunks: List[str],
        subsubfolder: Path
    ) -> None:
        safetensors.torch.save_file({"embeddings": embedding}, subsubfolder / f"embeddings.safetensors")
        with open(subsubfolder / f"ids.json", "w") as f:
            json.dump({"ids": ids}, f)
        with open(subsubfolder / f"documents.json", "w") as f:
            json.dump({"documents": documents}, f)
        with open(subsubfolder / f"chunks.json", "w") as f:
            json.dump({"chunks": chunks}, f)
    
    def ingest(self) -> None:
        """
        Store outputs into a folder structure like:
        /<you should name your folder's parent>
            /<you should create a folder with the dataset's name here>
                /<corpus | query>
                    /<model_name>
                        /embeddings.safetensors
                        /ids.jsonl
                        /documents.jsonl
        """
        # Each of these is a list
        for transformer_names in self.transformer_names:
            assert isinstance(transformer_names, list)
            print("WILL BE LOOKING AT TRANSFORMERS:", transformer_names)
            assert self.device is not None and self.device != "cpu" # lol
            print("CREATING TRANSFORMERS")
            transformers = [SentenceTransformer(t, device=self.device, token=os.environ.get("HF_ACCESS_TOKEN")) for t in tqdm(transformer_names, desc="Creating transformers")]
            print("---------------------------------------- PROCESSING JSONL FILES ----------------------------------------")
            # TODO(Adriano) parallel plz
            for transformer, transformer_name in zip(transformers, transformer_names):
                for jsonl_file in tqdm(list(self.datasets_path.glob("**/*.jsonl")), desc="Processing jsonl files"):
                    _rel = jsonl_file.parent.relative_to(self.datasets_path)
                    assert transformer_name not in _rel.as_posix()
                    target_dir = self.output_dir / _rel / transformer_name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    with open(jsonl_file, "r") as f:
                        lines = f.readlines()
                        _chunks = [Chunk.model_validate_json(line) for line in lines]
                        chunks = [c.text for c in _chunks]
                    assert isinstance(chunks, list)
                    assert all(isinstance(c, str) for c in chunks)
                    batched_chunks: List[List[str]] = [chunks[i:i+self.batch_size] for i in range(0, len(chunks), self.batch_size)]
                    assert isinstance(batched_chunks, list)
                    assert all(isinstance(b, list) for b in batched_chunks)
                    assert all(isinstance(c, str) for b in batched_chunks for c in b)
                    all_embeddings: List[torch.Tensor] = []
                    for batch in tqdm(batched_chunks, desc="Embedding chunks"):
                        batch_embeddings = transformer.encode(batch, normalize_embeddings=self.normalize_embeddings, convert_to_tensor=True)
                        assert isinstance(batch_embeddings, torch.Tensor)
                        assert len(batch_embeddings.shape) == 2
                        assert batch_embeddings.shape[0] == len(batch)
                        assert batch_embeddings.shape[1] == model2model_dimension(transformer_name)
                        all_embeddings.append(batch_embeddings.detach().cpu())
                    all_embeddings_pt = torch.cat(all_embeddings, dim=0)
                    assert len(all_embeddings_pt.shape) == 2
                    assert all_embeddings_pt.shape[0] == len(chunks)
                    assert all_embeddings_pt.shape[1] == model2model_dimension(transformer_name)
                    # 1. Save embeddings
                    safetensors.torch.save_file({"embeddings": all_embeddings_pt}, target_dir / f"embeddings.safetensors")
                    # 2. Save metadatas (just copy over)
                    with open(jsonl_file, "r") as f1:
                        with open(target_dir / f"metadatas.jsonl", "w") as f2:
                            f2.write(f1.read())
            for transformer in transformers:
                transformer.cpu()
            del transformers
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

@current_app.cli.command("ingest_hf")
@click.option("--output", "-o", type=click.Path(), default=Path(__file__).parent.parent.parent / "data" / "embeddings")
@click.option("--device", "-d", default="cuda:0")
@click.option("--parallelize", "-p", is_flag=True, default=False)
@click.option("--dataset", "-da", default="scidocs")
def ingest_hf(output: str, device: str, parallelize: bool, dataset: str):
    """
    Commands to run (probably)
    `flask ingest_hf -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings_huggingface/arguana -d cuda:0`
    `flask ingest_hf -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings_huggingface/fiqa -d cuda:1`
    `flask ingest_hf -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings_huggingface/scidocs -d cuda:2`
    `flask ingest_hf -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings_huggingface/nfcorpus -d cuda:3`
    """
    assert os.environ.get("HF_ACCESS_TOKEN") is not None
    # NOTE: not parallel since it's honestly easier to just run manually one per gpu
    assert dataset in DATASETS
    start_time = time.time()
    click.echo(f"Output: {output}")
    click.echo(f"Device: {device}")
    click.echo("========== INITIALIZING INGESTOR ==========")
    injestor = SingletonInjestor(Path(output), device, parallelize, dataset)
    click.echo("========== INGESTING DATASET ==========")
    injestor.ingest()
    click.echo("========== DONE INGESTING ==========")
    print("time taken:", time.time() - start_time)

# if __name__ == "__main__":
#     ingest_hf()

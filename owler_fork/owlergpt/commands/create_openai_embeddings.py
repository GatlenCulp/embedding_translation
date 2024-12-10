from __future__ import annotations
import torch
import click
import tqdm
import os
import safetensors.torch
from flask import current_app
import pydantic
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from openai import OpenAI
from langchain.text_splitter import TokenTextSplitter
import re

# ...
from owlergpt.utils import JSONDataset, collate_fn

MODEL_NAMES = ["text-embedding-3-small", "text-embedding-3-large"]
DATASETS = ["scidocs", "arguana", "fiqa", "hotpotqa", "nfcorpus", "scidocs", "trec-covid"]

class RecordMetadata(pydantic.BaseModel):
    record_id: str
    record_text: str
    record_type: str
    
def model_name2dim(model_name: str) -> int:
    if model_name == "text-embedding-3-large":
        return 3072
    elif model_name == "text-embedding-3-small":
        return 1536
    else:
        raise ValueError(f"Unknown model name: {model_name}")


@current_app.cli.command("ingest_openai")
@click.option("--dataset_path", "-d", type=str)
@click.option("--output_path", "-o", type=str, default="/mnt/align3_drive/adrianoh/dl_final_project_embeddings_openai")
@click.option("--regex-datasets", "-re", "-r", type=str, default=".*") # filter all datasets with a regex
@click.option("--chunks-path", "-c", type=str, default=None, help="Pass the chunks path to be able to just load chunks and work on those instead.")
def ingest_openai(dataset_path: Optional[str], output_path: Optional[str], regex_datasets: Optional[str], chunks_path: Optional[str]):
    print("Parsing arguments")
    if dataset_path is None:
        dataset_path = os.environ.get("DATASET_FOLDER_PATH")
    dataset_path = Path(dataset_path)
    assert chunks_path is not None # we must use chunks
    chunks_path = Path(chunks_path)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Creating OpenAI client")
    client = OpenAI(api_key=os.environ["OPENAI_KEY"])

    print("Fetching environment variables (defaults)")
    tokens_per_chunk = int(os.environ.get("CHUNK_SIZE", None))
    chunk_overlap = int(os.environ.get("VECTOR_SEARCH_TEXT_SPLITTER_CHUNK_OVERLAP", None))
    batch_size = int(os.environ.get("BATCH_SIZE", None))
    chunk_document_prefix = os.environ.get("VECTOR_SEARCH_CHUNK_PREFIX", None)
    chunk_query_prefix = os.environ.get("VECTOR_SEARCH_QUERY_PREFIX", None)
    assert chunk_document_prefix is not None
    assert chunk_query_prefix is not None
    
    dataset_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    print("Found dataset folders:")
    print("  " + "  \n".join([d.as_posix() for d in dataset_folders]))
    dataset_folders = [d for d in dataset_folders if re.match(regex_datasets, d.name) is not None]
    print("Filtered dataset folder (with regex):")
    print("  " + "  \n".join([d.as_posix() for d in dataset_folders]))
    for dataset_folder in dataset_folders:
        #### MAKE SURE ALL THE CHUNK FILES ARE THERE ####
        if chunks_path is not None:
            assert chunks_path / dataset_folder.name / "corpus.jsonl"
        assert not (output_path / dataset_folder.name).exists() or (len(list((output_path / dataset_folder.name).glob("*"))) == 0), f"Dataset folder {dataset_folder.as_posix()} already exists"
        (output_path / dataset_folder.name).mkdir(parents=True, exist_ok=True)
    chunk_folders = [chunks_path / d.name for d in dataset_folders]
    print("Will go through dataset folders: ")
    print("  " + "  \n".join([d.as_posix() for d in chunk_folders]))
    click.confirm("Continue?", abort=True)
    #### FOR EACH MODEL ####
    for model_name in MODEL_NAMES:
        print(f"=> Processing model: {model_name}")
        model_path = output_path / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        # text_splitter = TokenTextSplitter(
        #     model_name=model_name,
        #     chunk_overlap=chunk_overlap,
        #     chunk_size=tokens_per_chunk
        # ) if chunks_path is None else None
        #### FOR EACH DATASET ####
        for dataset_folder in chunk_folders:
            jsonl_files = list(dataset_folder.glob("**/*.jsonl"))
            assert all(f.is_file() for f in jsonl_files)
            for jsonl_file in tqdm.tqdm(jsonl_files, desc="Processing JSONL files from all of the documents..."):
                # 1. Make sure folders are OK and set up to write
                _rel = jsonl_file.parent.relative_to(dataset_folder.parent)
                _new_parent = model_path / _rel
                _new_parent.mkdir(parents=True, exist_ok=True)
                embeddings_file_path = _new_parent / f"{jsonl_file.name.replace('.jsonl', '_embeddings.safetensors')}"
                metadatas_file_path = _new_parent / f"{jsonl_file.name.replace('.jsonl', '_metadatas.jsonl')}"
                # [START DEBUG]
                # print("\n\nNEW PARENT IS", _new_parent.as_posix(), "\n\nREL IS", _rel.as_posix(), "\n\nJSONL FILE IS", jsonl_file.as_posix(), "\n\nMODEL PATH IS", model_path.as_posix(), "\n\nDATASET FOLDER IS", dataset_folder.as_posix(), "\n\n")
                # print("\n\nNEW EMBEDDINGS FILE PATH IS", embeddings_file_path.as_posix(), "\n\nNEW METADATAS FILE PATH IS", metadatas_file_path.as_posix(), "\n\n")
                # [END DEBUG]
                if embeddings_file_path.exists():
                    assert metadatas_file_path.exists()
                    continue
                else:
                    assert not embeddings_file_path.exists()
                    assert not metadatas_file_path.exists()
                    # 2. Set up datasets and splitters, etc...
                    record_type = "document" if "corpus" in jsonl_file.name else "query"
                    assert record_type == ("query" if "queries" in jsonl_file.name else "document")
                    # dataset = JSONDataset(
                    #     path=jsonl_file.as_posix(),
                    #     splitter=text_splitter, # Is None if chunks path is provided
                    #     model_name=model_name,
                    #     chunk_size=tokens_per_chunk,
                    #     chunk_overlap=int(os.environ.get("VECTOR_SEARCH_TEXT_SPLITTER_CHUNK_OVERLAP", "20")),
                    #     chunk_prefix=os.environ.get("VECTOR_SEARCH_CHUNK_PREFIX"),
                    #     record_type=record_type
                    # ) if chunks_path is None else None
                    # 3. Compute embeddings
                    embeddings = ingest_openai_embeddings_as_pt(
                        client,
                        jsonl_file,
                        # dataset, # None if chunks path is provided
                        batch_size,
                        model_name,
                        # record_type,
                        # is_chunk=(chunks_path is not None) # always true
                    )
                    # 4. Save embeddings
                    safetensors.torch.save_file({"embeddings": embeddings}, embeddings_file_path)
                    # 5. Copy over the jsonl file (metadatas)
                    with open(jsonl_file, 'r') as f1:
                        with open(metadatas_file_path, 'w') as f2:
                            f2.write(f1.read())

# NOTE: copied from chunk_dataset.py
class Chunk(BaseModel):
    id: str = Field(alias="id") # unique id for the chunk (can combine doc id with index within that chunk)
    doc_id: str = Field(alias="doc_id")
    index_in_doc: int = Field(alias="index_in_doc")
    text: str = Field(alias="text")

def ingest_openai_embeddings_as_pt(
        client: OpenAI,
        chunks_file_path: Path,
        # dataset: Optional[JSONDataset],
        batch_size: int,
        model_name: str,
        # record_type: str,
        # is_chunk: bool = False
    ) -> torch.Tensor:
    with open(chunks_file_path, 'r') as f:
        jsons = [Chunk.model_validate_json(line) for line in f]
        chunks = [j.text for j in jsons]
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert len(chunks) > 0, f"No chunks found for {chunks_file_path.as_posix()}"

    all_embeddings: List[torch.Tensor] = []
    for idx in tqdm.trange(0, len(chunks), batch_size):
        batch = chunks[idx:idx+batch_size]
        assert isinstance(batch, list)
        assert all(isinstance(chunk, str) for chunk in batch)
        embeddings_data = client.embeddings.create(input=batch, model=model_name).data
        embeddings = [e.embedding for e in embeddings_data]
        embeddings_pt = torch.tensor(embeddings, device='cpu')
        assert len(embeddings_pt.shape) == 2
        assert embeddings_pt.shape[0] == len(batch)
        assert embeddings_pt.shape[1] == model_name2dim(model_name)
        all_embeddings.append(embeddings_pt)
    assert len(all_embeddings) > 0, f"No embeddings found for {chunks_file_path.as_posix()}"
    embeddings_pt = torch.cat(all_embeddings, dim=0)
    assert len(embeddings_pt.shape) == 2
    assert embeddings_pt.shape[0] == len(chunks)
    assert embeddings_pt.shape[1] == model_name2dim(model_name)
    return embeddings_pt
    # TODO(Adriano) chunking?
    # dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
    # all_embeddings: List[torch.Tensor] = []
    # all_metadatas: List[RecordMetadata] = []
    # for documents, ids, text_chunks in tqdm.tqdm(dataloader, desc='| Computing embeddings |', total=len(dataloader)):
    #     if len(documents) == 0 or len(ids) == 0 or len(text_chunks) == 0:
    #         continue
    #     # Generate embeddings for each chunk using OpenAI
    #     data = client.embeddings.create(input=text_chunks, model=model_name).data
    #     embeddings = [entry.embedding for entry in data]
    #     assert isinstance(embeddings, list)
    #     assert isinstance(embeddings[0], list)
    #     assert isinstance(embeddings[0][0], float)
    #     embeddings_pt = torch.tensor(embeddings, device='cpu')
    #     all_embeddings.append(embeddings_pt)

    #     # Prepare metadata for each chunk
    #     metadatas = [
    #         RecordMetadata(record_id=ids[i], record_text=text_chunks[i], record_type=record_type)
    #         for i in range(len(text_chunks))
    #     ]
    #     all_metadatas.extend(metadatas)

    # embeddings_full_tensor = torch.cat(all_embeddings, dim=0)
    # assert len(embeddings_full_tensor.shape) == 2
    # assert len(all_metadatas) == embeddings_full_tensor.shape[0]
    # assert embeddings_full_tensor.shape[1] == model_name2dim(model_name)
    
    # return embeddings_full_tensor

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
def ingest_openai(dataset_path: Optional[str], output_path: Optional[str], regex_datasets: Optional[str]):
    print("Parsing arguments")
    if dataset_path is None:
        dataset_path = os.environ.get("DATASET_FOLDER_PATH")
    assert dataset_path is not None
    dataset_path = Path(dataset_path)

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
        assert not (output_path / dataset_folder.name).exists() or (len(list((output_path / dataset_folder.name).glob("*"))) == 0), f"Dataset folder {dataset_folder.as_posix()} already exists"
        (output_path / dataset_folder.name).mkdir(parents=True, exist_ok=True)
    print("Will go through dataset folders: ")
    print("  " + "  \n".join([d.as_posix() for d in dataset_folders]))
    #### FOR EACH DATASET ####
    iter_num = 0
    total_iters = len(dataset_folders) * len(MODEL_NAMES) * 2
    for dataset_folder in dataset_folders:
        dataset_name = dataset_folder.name
        assert dataset_name in DATASETS
        dataset_folder_path = os.path.join(dataset_path, dataset_folder)
        print(f"> Processing dataset folder: {dataset_folder_path}")

        #### FOR EACH MODEL ####
        for model_name in MODEL_NAMES:
            print(f" => Processing model: {model_name}")

            output_dir = output_path / dataset_name / model_name
            assert not output_dir.exists()
            output_dir.mkdir(parents=True, exist_ok=True)

            #### FOR EACH RECORD TYPE ####
            for record_type in ['corpus', 'query']:
                iter_num += 1
                print(f"  ===> Processing record type: {record_type} (this is {iter_num} / {total_iters})")
                filename = "corpus.jsonl" if record_type == 'corpus' else "queries.jsonl"
                file = dataset_folder / filename
                assert file.exists()
                print(" =====> Creating text splitter + Dataset")
                text_splitter = TokenTextSplitter(
                    model_name=model_name,
                    chunk_overlap=chunk_overlap,
                    chunk_size=tokens_per_chunk
                )
                dataset = JSONDataset(
                    path=file.as_posix(),
                    splitter=text_splitter,
                    model_name=model_name,
                    chunk_size=tokens_per_chunk,
                    chunk_overlap=int(os.environ.get("VECTOR_SEARCH_TEXT_SPLITTER_CHUNK_OVERLAP", "20")),
                    chunk_prefix=os.environ.get("VECTOR_SEARCH_CHUNK_PREFIX"),
                    record_type=record_type
                )
                print(" =====> Computing embeddings")
                embeddings, metadatas = ingest_openai_embeddings_as_pt(client, dataset, batch_size, model_name, record_type)
                print(" =====> Saving embeddings")
                output_file_embeddings = output_dir / f"{record_type}_embeddings.safetensors"
                output_file_metadatas = output_dir / f"{record_type}_metadatas.jsonl"
                assert not output_file_embeddings.exists()
                assert not output_file_metadatas.exists()
                safetensors.torch.save_file({"embeddings": embeddings}, output_file_embeddings)
                print(f"    Saved embeddings to {output_file_embeddings}")
                with open(output_file_metadatas, "w") as f:
                    for metadata in metadatas:
                        dump = metadata.model_dump_json()
                        assert "\n" not in dump # <----- should be escaped and one line per json
                        f.write(dump + "\n")
                print(f" <===== Saved metadatas to {output_file_metadatas}")

def ingest_openai_embeddings_as_pt(
        client: OpenAI,
        dataset: JSONDataset,
        batch_size: int,
        model_name: str,
        record_type: str
    ) -> torch.Tensor:
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
    all_embeddings: List[torch.Tensor] = []
    all_metadatas: List[RecordMetadata] = []
    for documents, ids, text_chunks in tqdm.tqdm(dataloader, desc='| Computing embeddings |', total=len(dataloader)):
        if len(documents) == 0 or len(ids) == 0 or len(text_chunks) == 0:
            continue
        # Generate embeddings for each chunk using OpenAI
        data = client.embeddings.create(input=text_chunks, model=model_name).data
        embeddings = [entry.embedding for entry in data]
        assert isinstance(embeddings, list)
        assert isinstance(embeddings[0], list)
        assert isinstance(embeddings[0][0], float)
        embeddings_pt = torch.tensor(embeddings, device='cpu')
        all_embeddings.append(embeddings_pt)

        # Prepare metadata for each chunk
        metadatas = [
            RecordMetadata(record_id=ids[i], record_text=text_chunks[i], record_type=record_type)
            for i in range(len(text_chunks))
        ]
        all_metadatas.extend(metadatas)

    embeddings_full_tensor = torch.cat(all_embeddings, dim=0)
    assert len(embeddings_full_tensor.shape) == 2
    assert len(all_metadatas) == embeddings_full_tensor.shape[0]
    assert embeddings_full_tensor.shape[1] == model_name2dim(model_name)
    
    return embeddings_full_tensor, all_metadatas

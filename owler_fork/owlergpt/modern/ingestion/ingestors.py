from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chromadb
import click
import cohere
from chromadb import PersistentClient
from chromadb import Settings
from chromadb.api.client import AdminClient
from chromadb.config import DEFAULT_TENANT
from chromadb.db.base import UniqueConstraintError
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.text_splitter import TokenTextSplitter
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm

from owlergpt.modern.collection_utils import COHERE_MODELS
from owlergpt.modern.collection_utils import OPENAI_MODELS
from owlergpt.utils import JSONDataset
from owlergpt.utils import collate_fn


# TODO(Adriano) also add a HF method? Maybe convert into HF dataset instead?
# Arguana example
# {
#   "_id": "test-environment-aeghhgwpe-pro02b",
#   "title": "animals environment general health health general weight philosophy ethics",
#   "text": "You don\u2019t have to be vegetarian to be green... Many special environments have been created by ...08"
# }
class StringsToJSONDataset:
    """Creates a JSONDataset from a list of strings by writing them to a JSONL file."""

    def __init__(self, output_path: str | Path):
        """Args:
        output_path: Path where the JSONL file will be written
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.queries_output_path = Path(output_path) / "queries.jsonl"
        self.corpus_output_path = Path(output_path) / "corpus.jsonl"

    def create_dataset(self, texts: list[str], queries: list[str]) -> None:
        """Creates a JSONL file from list of strings that can be loaded by JSONDataset.

        Args:
            texts: List of text strings to convert
            record_type: Type of records ("document" or "query")
        """
        assert (
            self.output_path.exists()
        ), f"Output path {self.output_path} does not exist"
        for output_path, records in [
            (self.queries_output_path, queries),
            (self.corpus_output_path, texts),
        ]:
            records: list[dict[str, Any]] = []
            for i, text in enumerate(texts):
                record = {
                    # Look at arguana example above, all MTEB datasets SEEM to be like this
                    "_id": f"text_{i}",
                    "title": f"text_{i}",
                    "text": text,
                }
                records.append(record)

            # Write records to JSONL file
            assert not output_path.exists(), f"Output path {output_path} already exists"
            assert (
                output_path.parent.exists()
            ), f"Output path parent {output_path.parent} does not exist"
            with open(output_path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")


class OriginalIngestion:
    """A static class to provide support for (a lot of) the original ingestion logic. This allows
    us to, for example, test. It also has methods to facilitate testing on fake datasets by
    turning hard-coded string datasets into JSONDataset datasets.
    """

    @staticmethod
    def __create_split_embedding_models(
        model_name: str, parallelism_batch_size: int = 1
    ) -> list[SentenceTransformer]:
        return [
            SentenceTransformer(
                model_name,
                device=os.environ["VECTOR_SEARCH_SENTENCE_TRANSFORMER_DEVICE"],
            )
            for _ in range(parallelism_batch_size)
        ]

    @staticmethod
    def __get_splitter_and_model(
        model_name: str,
        chunk_overlap: int,
        tokens_per_chunk: int,
        # none => get from environ
        openai_key: str | None = None,
        cohere_key: str | None = None,
    ) -> tuple[TokenTextSplitter, str, Any]:
        """Helper to acquire the (text_splitter, transformer_model, client) tuple from a model name and other parameters."""
        text_splitter, transformer_model, client = None, None, None
        if model_name in OPENAI_MODELS:
            text_splitter = TokenTextSplitter(
                model_name=model_name,
                chunk_overlap=chunk_overlap,
                chunk_size=tokens_per_chunk,
            )
            transformer_model = model_name
            if openai_key is None:
                if "OPENAI_KEY" not in os.environ:
                    raise ValueError(
                        "OPENAI_KEY is not set, you should set it since parallel inference requires GPU support"
                    )
                openai_key = os.environ["OPENAI_KEY"]
            client = OpenAI(api_key=openai_key)
        elif model_name in COHERE_MODELS:
            if cohere_key is None:
                if "COHERE_KEY" not in os.environ:
                    raise ValueError(
                        "COHERE_KEY is not set, you should set it since parallel inference requires GPU support"
                    )
                cohere_key = os.environ["COHERE_KEY"]
            client = cohere.Client(cohere_key)
            text_splitter = client
            transformer_model = model_name
        else:
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                raise ValueError(
                    "CUDA_VISIBLE_DEVICES is not set, you should set it since parallel inference requires GPU support"
                )
            text_splitter = SentenceTransformersTokenTextSplitter(
                model_name=model_name,
                chunk_overlap=chunk_overlap,
                tokens_per_chunk=tokens_per_chunk,  # Use the user-provided value
            )
            transformer_model = model_name.split("/")[-1]
        return text_splitter, transformer_model, client

    @staticmethod
    def create_collection(
        chroma_client: PersistentClient | None,
        vector_dataset_path: str,
        selected_folders: list[str],
        tokens_per_chunk: int,
        chunk_overlap: int,
        normalize_embeddings: bool,
        model_name: str,
        batch_size: int,
        dataset_folder_path: str,
        vector_search_chunk_prefix: str,
        vector_search_distance_function: str,
        log: bool = True,
        # none => get from environ
        openai_key: str | None = None,
        cohere_key: str | None = None,
        num_workers: int = 4,
    ) -> tuple[PersistentClient, chromadb.Collection]:
        if log:
            print("Getting text_splitter, transformer_model, client")
        text_splitter, transformer_model, client = (
            OriginalIngestion.__get_splitter_and_model(
                # fetch keys from environ
                model_name,
                chunk_overlap,
                tokens_per_chunk,
                openai_key,
                cohere_key,
            )
        )

        if log:
            print("Creating split embedding models")
        embedding_model = OriginalIngestion.__create_split_embedding_models(
            model_name, parallelism_batch_size=1
        )[0]

        tqdm_func = tqdm if log else lambda *args, **kwargs: args[0]
        # 0. Create Chroma Client
        if log:
            print("Creating collection")
        if chroma_client is None:
            chroma_client = PersistentClient(
                path=vector_dataset_path,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            # wtf? "._identifier?" cmon man, it came from here
            # ```
            # In [4]: chroma_client.__dict__
            # Out[4]:
            # {'_identifier': './hi',
            # 'tenant': 'default_tenant',
            # 'database': 'default_database',
            # '_admin_client': <chromadb.api.client.AdminClient at 0x125a90050>,
            # '_server': <chromadb.api.segment.SegmentAPI at 0x11586e6d0>}
            # ```
            assert chroma_client._identifier == vector_dataset_path
        admin_client = AdminClient.from_system(chroma_client._system)

        # 1. Create databases
        if log:
            print(f"Creating {len(selected_folders)} databases")
        db_names = [
            f"{selected_folder}_{tokens_per_chunk}"
            for selected_folder in selected_folders
        ]
        assert len(db_names) == len(selected_folders)
        collections = []
        for db_name, selected_folder in tqdm_func(
            zip(db_names, selected_folders, strict=False),
            desc="| Creating databases + collections |",
        ):
            try:
                admin_client.create_database(db_name)
                if log:
                    click.echo(
                        f"Created dataset-specific DB {db_name} to store embeddings."
                    )
            except UniqueConstraintError:
                if log:
                    click.echo(
                        f"Dataset-specific DB {db_name} already exists. Using it to store embeddings"
                    )
            chroma_client.set_tenant(tenant=DEFAULT_TENANT, database=db_name)
            # Include VECTOR_SEARCH_SENTENCE_TRANSFORMER_MODEL in the collection name
            collection_name = f"{selected_folder}_{transformer_model}_CharacterSplitting_{tokens_per_chunk}"

            try:
                # Attempt to create a new collection with the selected folder name
                chroma_collection = chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": vector_search_distance_function},
                )
                collections.append(chroma_collection)
            except UniqueConstraintError:
                # If the collection already exists, delete it and create a new one
                click.echo(
                    f"Collection {collection_name} already exists. Removing and creating a new one."
                )
                chroma_client.delete_collection(name=collection_name)
                chroma_collection = chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": vector_search_distance_function},
                )
                collections.append(chroma_collection)
        assert len(collections) == len(db_names)

        print(f"Processing dataset {selected_folders[0]}")
        total_records = 0
        total_embeddings = 0

        # Process the batch of documents
        # NOTE: single
        json_dataset_path = os.path.join(dataset_folder_path, selected_folders[0])
        if log:
            print(f"Processing dataset {json_dataset_path}")
        for filename in ["corpus.jsonl", "queries.jsonl"]:
            if filename == "queries.jsonl":
                record_type = "query"  # <---- imporant NOTE
            else:
                record_type = "document"
            dataset = JSONDataset(
                (Path(json_dataset_path) / filename).as_posix(),
                text_splitter,
                model_name,
                tokens_per_chunk,
                chunk_overlap,
                vector_search_chunk_prefix,
                record_type,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )
            total_records += dataset.__len__()
            for documents, ids, text_chunks in tqdm(
                dataloader, desc="| Computing embeddings |", total=len(dataloader)
            ):
                if len(documents) == 0 or len(ids) == 0 or len(text_chunks) == 0:
                    continue
                # Generate embeddings for each chunk
                if model_name in OPENAI_MODELS:
                    embeddings = []
                    data = client.embeddings.create(
                        input=text_chunks, model=model_name
                    ).data
                    for entry in data:
                        embeddings.append(entry.embedding)
                elif model_name in COHERE_MODELS:
                    embeddings = client.embed(
                        texts=text_chunks,
                        model=model_name,
                        input_type="search_" + record_type,
                        embedding_types=["float"],
                    ).embeddings.float
                else:
                    embeddings = embedding_model.encode(
                        text_chunks, normalize_embeddings=normalize_embeddings
                    ).tolist()

                # Prepare metadata for each chunk
                metadatas = [
                    {
                        "record_id": ids[i],
                        "record_text": text_chunks[i],
                        "record_type": record_type,
                    }
                    for i in range(len(text_chunks))
                ]

                total_embeddings += len(embeddings)

                # Store embeddings and metadata in the vector store
                chroma_collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )

            click.echo(
                f"Processed {total_records} documents, generated {total_embeddings} embeddings."
            )
        return chroma_client, collections

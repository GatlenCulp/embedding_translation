import json
import os
from pathlib import Path

import click
from flask import current_app
from langchain.text_splitter import TokenTextSplitter
from pydantic import BaseModel
from pydantic import Field
from tqdm import tqdm


def valid_datasets_folder(dataset_path: Path):
    files = list(dataset_path.glob("**/*.jsonl"))
    assert len(files) > 0 and len(files) % 2 == 0 and len([f for f in files if "corpus" in f.name]) == len(files) / 2 and all("corpus" in f.name or "queries" in f.name for f in files)  # fmt: skip
    _parents = set(f.parent for f in files)
    for parent in _parents:
        assert len(list(parent.glob("**/*.jsonl"))) in [4, 6]
        assert (parent / "corpus_train.jsonl").exists() and (
            parent / "queries_train.jsonl"
        ).exists()
        assert (parent / "corpus_validation.jsonl").exists() and (
            parent / "queries_validation.jsonl"
        ).exists()


def validate_all_ids_unique(dataset_path: Path, enforce_intra: bool = False):
    """Ensure intra-dataset and inter-dataset uniqueness of ids."""
    set_of_all_ids = set()
    for file in tqdm(
        list(dataset_path.glob("**/*.jsonl")), desc="Validating all ids are unique"
    ):
        with open(file) as f:
            lines = f.readlines()
        jsons = [json.loads(line) for line in lines]
        assert all("_id" in j for j in jsons)
        ids = {j["_id"] for j in jsons}
        assert len(ids) == len(jsons)
        if enforce_intra:
            assert not any(j["_id"] in set_of_all_ids for j in jsons)
            set_of_all_ids |= ids


class Chunk(BaseModel):
    id: str = Field(
        alias="id"
    )  # unique id for the chunk (can combine doc id with index within that chunk)
    doc_id: str = Field(alias="doc_id")
    index_in_doc: int = Field(alias="index_in_doc")
    text: str = Field(alias="text")


# NOTE: examples/samples are
@current_app.cli.command("chunk_dataset")
@click.option("--enforce-intra", "-i", is_flag=True, default=False)
def chunk_dataset(enforce_intra: bool):
    """Chunk each dataset into chunks and put them into the `chunks_datasets` folder."""
    # 0. Setup/parameters
    dataset_path = Path(os.environ["DATASET_FOLDER_PATH"])
    assert dataset_path.exists() and dataset_path.is_dir()
    print("Validating dataset folder...")
    valid_datasets_folder(dataset_path)

    print("Validating all ids are unique...")
    validate_all_ids_unique(dataset_path, enforce_intra=enforce_intra)

    # 1. Split everything into chunks in a canonical wa
    canonical_splitter = TokenTextSplitter(
        model_name="text-embedding-3-small",
        chunk_overlap=int(os.environ["VECTOR_SEARCH_TEXT_SPLITTER_CHUNK_OVERLAP"]),
        chunk_size=int(os.environ["CHUNK_SIZE"]),
    )

    output_directory = Path("./chunks")
    assert not output_directory.exists()
    output_directory.mkdir(parents=True, exist_ok=True)

    # 2. Process each jsonl file
    for file in tqdm(list(dataset_path.glob("**/*.jsonl")), desc="Chunking datasets"):
        # Create corresponding output file path preserving directory structure
        rel_path = file.relative_to(dataset_path)
        out_file = output_directory / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Read and process each line
        with open(file) as f:
            lines = f.readlines()
        assert len(lines) > 0, f"No lines found for {file.as_posix()}"

        chunked_documents: list[str] = []
        for line in tqdm(lines, desc=f"Processing {file.name}", leave=False):
            doc = json.loads(line)

            # Split text into chunks
            chunks: list[str] = canonical_splitter.split_text(doc["text"])

            # Create new documents for each chunk
            for i, chunk in enumerate(chunks):
                # NOTE _id -> id cuz otherwise pydantic yells at me
                chunk = Chunk(
                    id=f"doc_{doc['_id']}_chunk_{i}",
                    doc_id=doc["_id"],
                    index_in_doc=i,
                    text=chunk,
                )
                chunk_json = chunk.model_dump_json()
                assert "\n" not in chunk_json
                chunked_documents.append(chunk_json)

        # Write chunked documents to output file
        assert len(chunked_documents) > 0, f"No chunks created for {file.as_posix()}"
        with open(out_file, "w") as f:
            for doc in chunked_documents:
                f.write(doc + "\n")

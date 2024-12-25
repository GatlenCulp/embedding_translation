from __future__ import annotations


"""
Helper by adrianoh to consolidate repeated functionality in CLI commands.
"""

import os

import chromadb
import click
from chromadb.config import DEFAULT_TENANT

from owlergpt.modern.collection_utils import MODEL_NAMES
from owlergpt.modern.collection_utils import parse_collection_name
from owlergpt.utils import choose_dataset_folders


def get_selected_folder(environ: dict) -> str:
    selected_folders = choose_dataset_folders(environ["DATASET_FOLDER_PATH"])
    if len(selected_folders) != 1:
        raise ValueError("Exactly one folder must be selected")
    if selected_folders is None:
        raise ValueError("No selected folders")
    return selected_folders[0]


def get_chroma_collections(
    chroma_client: chromadb.PersistentClient,
    selected_folder: str,
    enforce_valid: bool = True,
) -> list[str]:
    """Return list of valid chroma collection names for a given client. You can later
    use any of these names to fetch that collection and do what you like. If you pass
    `enforce_valid` as `True`, it will check that for your given dataset (`selected_folder`)
    every single model from the supported models has a collection.
    """
    try:
        db_name = selected_folder + "_" + os.environ.get("CHUNK_SIZE")
        chroma_client.set_tenant(tenant=DEFAULT_TENANT, database=db_name)
    except ValueError:
        click.echo("No separate database found for dataset. Using default database.")

    # Fetch and list all collections
    collections = chroma_client.list_collections()
    if not collections:
        click.echo("No collections found.")
        return None
    collections = [c.name for c in collections]
    collections.sort()

    if enforce_valid:
        collections_dump_string = "\n".join(collections)
        assert len(collections) == len(MODEL_NAMES), f"Expected {len(MODEL_NAMES)} collections, got {len(collections)}\n{collections_dump_string}"  # fmt: skip
        coll_model_names = [
            parse_collection_name(collection)[1] for collection in collections
        ]
        sums_to_be_1: list[tuple[str, bool]] = []
        for c_model_name in coll_model_names:
            # O(n^2) is ok... (small)
            sums_to_be_1.append(
                (
                    c_model_name,
                    sum(c_model_name in model_name for model_name in MODEL_NAMES) == 1,
                )
            )
        sums_to_be_1_dump = "\n".join(
            f"{c_model_name}: {x}" for c_model_name, x in sums_to_be_1
        )
        assert all(x for _, x in sums_to_be_1), sums_to_be_1_dump

    return collections

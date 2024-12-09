from __future__ import annotations

"""
Helper by adrianoh to consolidate repeated functionality in CLI commands.
"""

import os
import click
import chromadb
from chromadb.config import DEFAULT_TENANT
from typing import List
from owlergpt.utils import choose_dataset_folders


def get_selected_folder(environ: dict) -> str:
    selected_folders = choose_dataset_folders(environ["DATASET_FOLDER_PATH"])
    if len(selected_folders) != 1:
        raise ValueError("Exactly one folder must be selected")
    if selected_folders is None:
        raise ValueError("No selected folders")
    return selected_folders[0]


def get_chroma_collections(
    chroma_client: chromadb.PersistentClient, selected_folder: str
) -> List[str]:
    try:
        db_name = selected_folder + "_" + os.environ.get("CHUNK_SIZE")
        chroma_client.set_tenant(tenant=DEFAULT_TENANT, database=db_name)
    except ValueError:
        click.echo("No separate database found for dataset. Using default database.")

    # Fetch and list all collections
    collections = chroma_client.list_collections()
    if not collections:
        click.echo("No collections found.")
        return
    collections = [c.name for c in collections]
    collections.sort()
    return collections

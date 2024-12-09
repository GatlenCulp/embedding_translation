import os
import click
import re
import requests
from typing import List
from pathlib import Path

from flask import current_app

# NOTE: all of these have a folder structure with these two we care about:
# - corpus.jsonl
# - queries.jsonl
DEFAULT_DATASETS = [
    "mteb/fiqa",
    "mteb/nfcorpus",
    "mteb/hotpotqa",
    "mteb/trec-covid",
    "mteb/scidocs",
    "mteb/arguana"
]

# could subproc. curl or chunking be faster/better?
def download_dataset_jsonl(dataset_name: str, files: List[str], folder: Path) -> None:
    download_link_template = "https://huggingface.co/datasets/{dataset_name}/resolve/main/{file}?download=true"
    download_links = [download_link_template.format(dataset_name=dataset_name, file=file) for file in files]
    for file, link in zip(files, download_links):
        click.echo(f"Downloading {link}...")
        response = requests.get(link, stream=True)
        response.raise_for_status()
        with open(folder / file, 'wb') as f:
            f.write(response.content)

@current_app.cli.command("download_ds")
def download_datasets() -> None:
    """
    Downloads datasets from the BEIR leaderboard.
    """
    _datasets_folder = os.environ["DATASET_FOLDER_PATH"]
    if not _datasets_folder:
        raise ValueError("DATASET_FOLDER_PATH is not set")
    datasets_folder = Path(_datasets_folder)
    for name in DEFAULT_DATASETS:
        _, folder_name = name.split("/", 1)
        assert all(re.match(r"^[a-zA-Z0-9_-]+$", part) for part in [_, folder_name])
        folder = datasets_folder / folder_name
        if folder.exists():
            click.echo(f"Folder {folder} already exists, skipping download")
            continue
        folder.mkdir(parents=True, exist_ok=False)
        click.echo(f"Downloading dataset {name} to {folder}")
        file_names = ["corpus.jsonl", "queries.jsonl"] # train set ok, no training happening
        download_dataset_jsonl(name, file_names, folder)

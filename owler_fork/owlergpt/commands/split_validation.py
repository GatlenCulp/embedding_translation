import os
import click
from pathlib import Path
import numpy as np
from flask import current_app
from tqdm import tqdm
def valid_datasets_folder(dataset_path: Path):
    files = list(dataset_path.glob("**/*.jsonl"))
    assert len(files) > 0 and len(files) % 2 == 0 and len([f for f in files if "corpus" in f.name]) == len(files) / 2 and all("corpus" in f.name or "queries" in f.name for f in files) # fmt: skip
    _parents = set(f.parent for f in files)
    for parent in _parents:
        assert len(list(parent.glob("**/*.jsonl"))) == 2
        assert (parent / "corpus.jsonl").exists() and (parent / "queries.jsonl").exists()

# NOTE 80/20 of 25K (our default max size) -> 20K train and 5K validation
@current_app.cli.command("split_validation")
@click.option("--validation-split", "-v", type=float, default=0.2)
@click.option("--delete-original-files", "-d", is_flag=True, default=False)
def split_validation(validation_split: float, delete_original_files: bool):
    """
    Go into the datasets folder, ensure that it's valid, and then go through every file, and split it into into train/validation sets.
    Specifically, every jsonl file which due tothe validation should be known to be called corpus.jsonl or queries.jsonl, should be
    moved into <name>_train.jsonl and <name>_validation.jsonl. Split the number of lines based on the validation split.
    """
    # 0. Setup/parameters
    dataset_path = Path(os.environ["DATASET_FOLDER_PATH"])
    assert dataset_path.exists() and dataset_path.is_dir()
    valid_datasets_folder(dataset_path)

    # 1. Split the validation set
    for file in tqdm(list(dataset_path.glob("**/*.jsonl")), desc="Splitting validation set"):
        assert file.stem in ["corpus", "queries"]
        train_file = file.parent / (file.stem + "_train.jsonl")
        validation_file = file.parent / (file.stem + "_validation.jsonl")
        assert not train_file.exists() and not validation_file.exists()
        with open(file, "r") as f:
            lines = f.readlines()
        np.random.shuffle(lines)
        num_train_lines = int(len(lines) * (1 - validation_split))
        num_validation_lines = len(lines) - num_train_lines
        assert len(lines) == num_train_lines + num_validation_lines
        assert num_train_lines > 0 and num_validation_lines > 0
        train_lines = lines[:num_train_lines]
        validation_lines = lines[num_train_lines:]
        with open(train_file, "w") as f:
            f.writelines(train_lines)
        with open(validation_file, "w") as f:
            f.writelines(validation_lines)
        if delete_original_files:
            file.unlink()


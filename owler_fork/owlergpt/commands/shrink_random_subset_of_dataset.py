import click
import numpy as np
from pathlib import Path
import os
from flask import current_app

# NOTE: examples/samples are 
@current_app.cli.command("shrink_ds")
@click.option("--max-num-examples", "-n", type=int, default=25_000)
def shrink(max_num_examples: int):
    # 0. Setup/parameters
    dataset_path = Path(os.environ["DATASET_FOLDER_PATH"] + "_old") # NOTE: before you run this mv the folder
    dataset_out_path = Path(os.environ["DATASET_FOLDER_PATH"])
    click.confirm(f"Moving from {dataset_path.as_posix()} to {dataset_out_path.as_posix()}", abort=True)
    assert not dataset_out_path.exists() or len(list(dataset_out_path.glob("*"))) == 0

    # 1. Validation checking TODO this is copied around thet place (inelegant)
    files = list(dataset_path.glob("**/*.jsonl"))
    assert len(files) > 0 and len(files) % 2 == 0 and len([f for f in files if "corpus" in f.name]) == len(files) / 2 and all("corpus" in f.name or "queries" in f.name for f in files) # fmt: skip
    _parents = set(f.parent for f in files)
    for parent in _parents:
        assert len(list(parent.glob("**/*.jsonl"))) == 2
        assert (parent / "corpus.jsonl").exists() and (parent / "queries.jsonl").exists()
    
    # 3. Shuffle and clip
    # Open it, shuffle the lines, then clip and re-write to a new file
    for file in files:
        rel_path = file.relative_to(dataset_path)
        out_file = dataset_out_path / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "r") as f:
            lines = f.readlines()
        np.random.shuffle(lines)
        lines = lines[:max_num_examples]
        with open(out_file, "w") as f:
            f.writelines(lines) # NOTE you must manually inspect this is a valid jsonl and looks OK
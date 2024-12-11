"""
COPY OF `train_on_safetensors_dbs.ipynb`
"""

# Define the necessary pydantic structures where we will store everything. Everything will be in:
#     <model1_folderable_name>_<model2_folderable_name>/
#         <dataset_name>/
#             # Linear transform
#             linear_transform.safetensors
#             stitch_info.json
#             # Embeddings (generated via linear(model1(texts))) where linear maps from model1 space to model2 space
#             embeddings_corpus_train.safetensors
#             embeddings_corpus_validation.safetensors
#             embeddings_query_train.safetensors
#             embeddings_query_validation.safetensors
#             # Metadatas
#             metadatas_corpus_train.json
#             metadatas_corpus_validation.json
#             metadatas_query_train.json
#             metadatas_query_validation.json
import torch.nn as nn
from pydantic import BaseModel
import itertools
import torch
import safetensors
import pydantic
from safetensors.torch import load_file
import safetensors.torch
from typing import Literal, Optional, List, Dict, Tuple
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import tqdm
import torch.optim as optim
from pydantic import BaseModel
import wandb
from typing import Optional
from pathlib import Path

class StitchPair(BaseModel):
    source: str
    target: str
    dataset: str
    mode: Literal["affine"] = "affine"  # TODO(Adriano) later we will support more shit here

    def save_linear_transform(self, linear: nn.Linear, save_path: Path) -> None:
        linear_path = save_path / "linear_transform.safetensors"
        stitch_info_path = save_path / "stitch_info.json"
        assert not linear_path.exists(), f"Linear transform already exists at {linear_path}"
        assert not stitch_info_path.exists(), f"Stitch info already exists at {stitch_info_path}"
        safetensors.torch.save_file(linear.state_dict(), linear_path)
        stitch_info_path.write_text(self.model_dump_json())

# Now, we can try to train linear transforms between embeddings...
class EmbeddingDataset(Dataset):
    def __init__(self, source_embeddings: torch.Tensor, target_embeddings: torch.Tensor):
        assert source_embeddings.shape[0] == target_embeddings.shape[0]
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings
        
    def __len__(self):
        return len(self.source_embeddings)
        
    def __getitem__(self, idx):
        return self.source_embeddings[idx], self.target_embeddings[idx]

class LinearTransformTrainerArgs(BaseModel):
    # We use default for now :)
    test_split: float = 0.2
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    save_every_n_epochs: int = 10
    use_tqdm: bool = True

class LinearTransformTrainer:
    def __init__(
        self,
        save_path: Path,
        linear: Optional[nn.Linear],
        source_embeddings_train: torch.Tensor,
        target_embeddings_train: torch.Tensor,
        source_embeddings_validation: torch.Tensor,
        target_embeddings_validation: torch.Tensor,
        device: torch.device | str,
        args: LinearTransformTrainerArgs = LinearTransformTrainerArgs()
    ):
        self.linear = linear
        self.source_embeddings_train = source_embeddings_train
        self.target_embeddings_train = target_embeddings_train
        self.source_embeddings_validation = source_embeddings_validation
        self.target_embeddings_validation = target_embeddings_validation
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.device = device
        self.test_split = args.test_split
        self.save_every_n_epochs = args.save_every_n_epochs
        self.save_path = save_path
        self.checkpoint_path = self.save_path / "checkpoints"
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.logfile = self.save_path / "log.jsonl"
        self.use_tqdm = args.use_tqdm
        if self.linear is None:
            self.linear = self.create_linear_transform()
        self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=self.learning_rate)

    #### PRE-TRAINING HELPERS ####
    def create_datasets(self):
        # Create indices and shuffle
        # num_samples = len(self.source_embeddings_train)
        # indices = torch.randperm(num_samples)
        
        # Split indices -> NOTE we already created this shit
        # split_idx = int(num_samples * (1 - self.test_split))
        # train_indices = indices[:split_idx]
        # test_indices = indices[split_idx:]
        
        # Create datasets
        train_dataset = EmbeddingDataset(
            self.source_embeddings_train,
            self.target_embeddings_train
        )
        test_dataset = EmbeddingDataset(
            self.source_embeddings_validation,
            self.target_embeddings_validation
        )
        
        return train_dataset, test_dataset

    def create_linear_transform(self):
        return nn.Linear(self.source_embeddings_train.shape[1], self.target_embeddings_train.shape[1]).to(self.device)

    #### TRAINING HELPERS ####
    def train(self):
        train_dataset, test_dataset = self.create_datasets()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        mse_loss = nn.MSELoss()
        trange = tqdm.trange if self.use_tqdm else range
        tqdm_fn = tqdm.tqdm if self.use_tqdm else lambda *args, **kwargs: args[0]
        for epoch in trange(self.num_epochs):
            # Training
            self.linear.train()
            train_mse = 0.0
            train_mae = 0.0
            num_train_batches = 0

            for source_emb, target_emb in train_loader:
                source_emb = source_emb.to(self.device)
                target_emb = target_emb.to(self.device)

                self.optimizer.zero_grad()
                output = self.linear(source_emb)

                loss = mse_loss(output, target_emb)
                loss.backward()
                self.optimizer.step()

                train_mse += loss.detach().item()
                train_mae += (output.detach() - target_emb.detach()).abs().mean().item()
                num_train_batches += 1

            avg_train_mse = train_mse / num_train_batches
            avg_train_mae = train_mae / num_train_batches

            # Evaluation
            self.linear.eval()
            test_mse = 0.0
            test_mae = 0.0
            num_test_batches = 0

            with torch.no_grad():
                for source_emb, target_emb in tqdm_fn(test_loader):
                    source_emb = source_emb.to(self.device)
                    target_emb = target_emb.to(self.device)

                    output = self.linear(source_emb)

                    test_mse += mse_loss(output, target_emb).item()
                    test_mae += (output.detach() - target_emb.detach()).abs().mean().item()
                    num_test_batches += 1

            avg_test_mse = test_mse / num_test_batches
            avg_test_mae = test_mae / num_test_batches

            # Log metrics
            log_entry = {
                "epoch": epoch,
                "train_mse": avg_train_mse,
                "train_mae": avg_train_mae,
                "test_mse": avg_test_mse,
                "test_mae": avg_test_mae,
            }
            wandb.log(log_entry)
            with open(self.logfile, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if epoch % self.save_every_n_epochs == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch: int):
        checkpoint_path = self.checkpoint_path / f"checkpoint_{epoch}.safetensors"
        safetensors.torch.save_file(self.linear.state_dict(), checkpoint_path)

DEFAULT_ARGS = LinearTransformTrainerArgs(
    test_split=0.2,
    num_epochs=50,
    batch_size=32,
    learning_rate=0.001,
    save_every_n_epochs=10,
    use_tqdm=True
)


DATASETS = [
    # (numbers are counts for documents, there may be some longer documents -> slightly more chunks)
    "arguana", # 10K
    "fiqa", # 50K -> 20K
    "scidocs", # 25K -> 20K
    "nfcorpus", # 5K
    "hotpotqa", # 100K -> 20K
    "trec-covid", # too much -> 20K
]

MODEL_NAMES = [
    "Salesforce/SFR-Embedding-Mistral",
    "WhereIsAI/UAE-Large-V1",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
    "intfloat/e5-small-v2",
    "thenlper/gte-base",
    "thenlper/gte-large",
    "thenlper/gte-small",
    "sentence-transformers/gtr-t5-base",
    "sentence-transformers/gtr-t5-large",
    "mixedbread-ai/mxbai-embed-large-v1",
    "sentence-transformers/sentence-t5-base",
    "sentence-transformers/sentence-t5-large",
    "text-embedding-3-large", # openai
    "text-embedding-3-small", # openai
]

class EmbeddingTransformTrainer:
    """Handles training transforms between embedding models."""
    
    def __init__(self, save_path_parent: Path, load_path_parent: Path, device: str, wandb_project: str):
        self.save_path_parent = save_path_parent
        self.load_path_parent = load_path_parent
        self.device = device
        self.wandb_project = wandb_project
        
    def train_all_pairs(
        self,
        filter_against_model: List[str] = [],
        filter_for_model: List[str] = [],
        filter_for_dataset: List[str] = [],
        filter_against_dataset: List[str] = []
    ):
        """Train transforms between all pairs of embeddings."""
        # filtering...
        combos = list(itertools.product(DATASETS, MODEL_NAMES, MODEL_NAMES))
        combos = [x for x in combos if x[1] != x[2]]
        # 1. filter for models
        for filter_against_model in filter_against_model:
            combos = [x for x in combos if filter_against_model not in x[1] and filter_against_model not in x[2]]
        for filter_for_model in filter_for_model:
            combos = [x for x in combos if filter_for_model in x[1] and filter_for_model in x[2]]
        # 2. filter for datasets
        for filter_for_dataset in filter_for_dataset:
            combos = [x for x in combos if filter_for_dataset in x[0]]
        for filter_against_dataset in filter_against_dataset:
            combos = [x for x in combos if filter_against_dataset not in x[0]]
        # End filtering...
        print(f"Training {len(combos)} transforms")
        for dataset, src, dest in tqdm.tqdm(combos, desc="Training transforms (all default settings)"):
            assert src != dest
            # plz plz
            src_name_ok = src.replace("/", "_")
            dest_name_ok = dest.replace("/", "_")
            dataset_ok = dataset.replace("/", "_")

            # 1. make sure shit is present in load
            embeddings1_path = self.load_path_parent / src_name_ok / dataset_ok
            embeddings2_path = self.load_path_parent / dest_name_ok / dataset_ok
            # NOTE ====> only corpus right now
            embeddings1_train_path = embeddings1_path / "embeddings_corpus_train.safetensors"
            embeddings1_validation_path = embeddings1_path / "embeddings_corpus_validation.safetensors"
            embeddings2_train_path = embeddings2_path / "embeddings_corpus_train.safetensors"
            embeddings2_validation_path = embeddings2_path / "embeddings_corpus_validation.safetensors"
            if any(not x.exists() for x in [embeddings1_train_path, embeddings1_validation_path, embeddings2_train_path, embeddings2_validation_path]):
                print(f"Skipping {src} to {dest} because some files do not exist in {embeddings1_path.name} or {embeddings2_path.name}")
                for file in [embeddings1_train_path, embeddings1_validation_path, embeddings2_train_path, embeddings2_validation_path]:
                    if file.exists():
                        print(f"File {file} exists")
                    else:
                        print(f"File {file} does not exist")
                continue

            # 2. make sure shit is NOT present in save
            save_parent = self.save_path_parent / f"{src_name_ok}_{dest_name_ok}" / dataset_ok
            stitch_info_pair_file = save_parent / "stitch_info_pairs.json"
            linear_transform_file = save_parent / "linear_transform.safetensors"
            assert not linear_transform_file.exists()
            assert not stitch_info_pair_file.exists()
            save_parent.mkdir(parents=True, exist_ok=True)
            stitch_info_pair = StitchPair(
                source=src,
                target=dest,
                dataset=dataset
            )
            stitch_info_pair_file.write_text(stitch_info_pair.model_dump_json())

            # 3. load
            embeddings1_train = load_file(embeddings1_train_path)["embeddings"] # X
            embeddings1_validation = load_file(embeddings1_validation_path)["embeddings"] # Y
            embeddings2_train = load_file(embeddings2_train_path)["embeddings"] # x
            embeddings2_validation = load_file(embeddings2_validation_path)["embeddings"] # y
            # Initialize wandb run
            wandb.init(
                project=self.wandb_project,
                name=f"{src_name_ok}_{dest_name_ok}",
                reinit=True
            )
            # Train the model
            trainer = LinearTransformTrainer(
                save_path=save_parent,
                linear=None,
                source_embeddings_train=embeddings1_train.to(self.device),
                target_embeddings_train=embeddings2_train.to(self.device),
                source_embeddings_validation=embeddings1_validation.to(self.device),
                target_embeddings_validation=embeddings2_validation.to(self.device),
                device=self.device,
                args=DEFAULT_ARGS
            )
            trainer.train()
            wandb.finish()
# RUN WITH
#
# python3 owlergpt/commands/cereal.py
if __name__ == "__main__":
    trainer = EmbeddingTransformTrainer(
        save_path_parent=Path("/mnt/align3_drive/adrianoh/dl_final_project_layers/arguana_hf_cartesian_product"),
        load_path_parent=Path("/mnt/align3_drive/adrianoh/dl_final_project_embeddings_huggingface"),
        device="cuda:0",
        wandb_project="2024_12_10_dl_project_layer_arguana_hf_only_train"
    )
    trainer.train_all_pairs(
        filter_against_model=["Salesforce/SFR-Embedding-Mistral","text-embedding-3-large","text-embedding-3-small"],
        filter_for_dataset=["arguana"],
    )

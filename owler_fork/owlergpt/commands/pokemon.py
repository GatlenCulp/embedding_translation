from __future__ import annotations
import click
import random
import math
import gc
import torch
import einops
import numpy as np
import time
import torch.nn as nn
from typing import List, Optional, Dict, Literal, Tuple
from pathlib import Path
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset, DataLoader
import tqdm

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
    "openai/text-embedding-3-large",
    "openai/text-embedding-3-small",
]

class StitchPair(BaseModel):
    source: str
    target: str
    dataset: str
    num_layers: Optional[int] = None
    # NOTE: first layer size SHOULD be input dim and last layer SHOULD size be output dim
    layer_dims: Optional[List[int]] = None
    mode: Literal["affine", "mlp"] = "mlp"

class MLP(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        source_model_name: str,
        target_model_name: str,
        dataset: str,
        hidden_dims: Optional[List[int]] = None,
        num_layers: int = 1
    ):
        super().__init__()
        num_layers = num_layers or len(hidden_dims)
        if num_layers is None:
            raise ValueError("Either num_layers or hidden_dims must be provided")
        if hidden_dims is None:
            hidden_dims = [max(input_dim, output_dim)] * (num_layers - 1)
            
        # Build layer dimensions including input and output
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        
        # Create sequential model with linear layers and ReLU activations
        layers: List[nn.Module] = []
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims)-2:  # No ReLU after final layer
                layers.append(nn.ReLU())
                
        self.model = nn.Sequential(*layers)
        self.source_model_name = source_model_name
        self.target_model_name = target_model_name
        self.dataset = dataset
        self.num_layers = num_layers
        self.layer_dims = layer_dims
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def save_to_folder_path(self, save_path: Path):
        assert not save_path.is_file(), f"Save path {save_path} is a file, not a directory"
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / f"mlp.safetensors"
        info_file = save_path / f"stitch_info.json"
        assert not model_file.exists(), f"Model file already exists at {model_file}"
        assert not info_file.exists(), f"Info file already exists at {info_file}"
        save_file(self.state_dict(), model_file)
        info_file.write_text(
            StitchPair(
                source=self.source_model_name, 
                target=self.target_model_name, 
                dataset=self.dataset, 
                mode="mlp", 
                num_layers=self.num_layers, 
                layer_dims=self.layer_dims
            ).model_dump_json()
        )
    
    def __repr__(self):
        return f"MLP(input_dim={self.input_dim}, output_dim={self.output_dim}, hidden_dims={self.hidden_dims}, num_layers={self.num_layers})"
    
class EmbeddingDataset(Dataset):
    def __init__(self, source_embeddings: torch.Tensor, target_embeddings: torch.Tensor):
        assert source_embeddings.shape[0] == target_embeddings.shape[0]
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings
        
    def __len__(self):
        return len(self.source_embeddings)
        
    def __getitem__(self, idx):
        return self.source_embeddings[idx], self.target_embeddings[idx]

def get_embeddings_paths(embeddings_path: Path):
    record_type = "corpus"
    embeddings_train_path = embeddings_path / f"embeddings_{record_type}_train.safetensors"
    embeddings_validation_path = embeddings_path / f"embeddings_{record_type}_validation.safetensors"
    assert (
        (embeddings_train_path.exists() and embeddings_validation_path.exists()) or 
        (not embeddings_train_path.exists() and not embeddings_validation_path.exists())
    )
    if not embeddings_train_path.exists():
        # NOTE: that sometimes the path names are reversed, i.e. when using OpenAI models; you can observe
        # more in detail in `get_reversed_model_files` in `sanity_check_embeddings_note_equal.ipynb`
        embeddings_train_path = embeddings_path / f"{record_type}_train_embeddings.safetensors"
        embeddings_validation_path = embeddings_path / f"{record_type}_validation_embeddings.safetensors"
    assert embeddings_train_path.exists() and embeddings_validation_path.exists(), f"Files {embeddings_train_path} and {embeddings_validation_path} do not exist" # fmt: skip
    return embeddings_train_path, embeddings_validation_path

def get_all_embeddings(embeddings_path: Path, dataset: str, device: str) -> Dict[str, torch.Tensor]:
    model2embeddings: Dict[str, torch.Tensor] = {}
    for model_name in MODEL_NAMES:
        embeddings_path_src_parent  = embeddings_path / model_name.replace("/", "_") / dataset
        embeddings_path_src_train, embeddings_path_src_validation = get_embeddings_paths(embeddings_path_src_parent)
        embeddings_src_train = load_file(embeddings_path_src_train)["embeddings"].to(device)
        model2embeddings[model_name] = embeddings_src_train
    return model2embeddings

def train_models(
        src_emb: torch.Tensor, 
        dst_emb: torch.Tensor, 
        models_src2dst: List[MLP], 
        models_dst2src: List[MLP], 
        optimizer: torch.optim.Optimizer, 
        loss_fn: nn.Module
    ) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = src_emb.shape[0]
    num_models = len(models_src2dst)
    
    # Expand embeddings to match number of models
    src_emb_expanded = einops.repeat(src_emb, 'b d -> n b d', n=num_models, b=batch_size) # [num_models, batch_size, dim]
    dst_emb_expanded = einops.repeat(dst_emb, 'b d -> n b d', n=num_models, b=batch_size) # [num_models, batch_size, dim]
    
    optimizer.zero_grad()
    
    # Forward pass (all models at once)
    outputs_src2dst = torch.stack([model(src_emb) for model in models_src2dst]) # [num_models, batch_size, dim]
    outputs_dst2src = torch.stack([model(dst_emb) for model in models_dst2src]) # [num_models, batch_size, dim]
    assert len(outputs_src2dst.shape) == len(outputs_dst2src.shape) == 3
    assert outputs_src2dst.shape[0] == outputs_dst2src.shape[0] == num_models
    assert outputs_src2dst.shape[1] == outputs_dst2src.shape[1] == batch_size
    # Compute loss for all models simultaneously
    loss_src2dst = loss_fn(outputs_src2dst, dst_emb_expanded)
    assert len(loss_src2dst.shape) == 3
    assert loss_src2dst.shape[0] == num_models
    assert loss_src2dst.shape[1] == batch_size
    assert loss_src2dst.shape[2] == dst_emb.shape[1]
    loss_src2dst = loss_src2dst.mean(dim=(1, 2))
    loss_src2dst_cpy = loss_src2dst.clone().detach().cpu().numpy()
    loss_src2dst = loss_src2dst.sum()
    loss_dst2src = loss_fn(outputs_dst2src, src_emb_expanded)
    assert len(loss_dst2src.shape) == 3
    assert loss_dst2src.shape[0] == num_models
    assert loss_dst2src.shape[1] == batch_size
    assert loss_dst2src.shape[2] == src_emb.shape[1]
    loss_dst2src = loss_dst2src.mean(dim=(1, 2))
    loss_dst2src_cpy = loss_dst2src.clone().detach().cpu().numpy()
    loss_dst2src = loss_dst2src.sum()
    loss = loss_src2dst + loss_dst2src
    
    # Backward pass and optimization
    loss.mean().backward()
    optimizer.step()
    
    return loss_src2dst_cpy, loss_dst2src_cpy

def training_run(
        datasets: List[EmbeddingDataset], 
        models_src2dst: List[List[MLP]], 
        models_dst2src: List[List[MLP]], 
        loss_fn: nn.Module, 
        num_epochs: int, 
        batch_size: int,
    ) -> float:
    time_start = time.time()
    train_loaders = [DataLoader(d, batch_size=batch_size, shuffle=True) for d in datasets]
    num_iters = len(train_loaders[0]) # NOTE: same for all of the datasets
    all_parameters = []
    # 1. Get all parameters
    for models_list in models_src2dst + models_dst2src:
        for models in models_list:
            # NOTE: tick together, but I think this is OK
            all_parameters.extend(list(models.parameters()))
    optimizer = torch.optim.Adam(all_parameters, lr=1e-3) # TODO(Adriano): should be ok to use defaults? lmao
    # 2. Set all models to train
    for models_list in models_src2dst + models_dst2src:
        for models in models_list:
            models.train()
    # 3. Train
    loss_fn = nn.MSELoss(reduction="none")
    for epoch in tqdm.trange(num_epochs):
        loader_iters = [iter(loader) for loader in train_loaders]
        for i in range(num_iters):
            xys = [next(loader_iter) for loader_iter in loader_iters]
            assert not any(x is None for x in xys), "Some loader iterators returned None"
            for (X, Y), models_src2dst_list, models_dst2src_list in zip(xys, models_src2dst, models_dst2src):
                losses_src2dst, losses_dst2src = train_models(X, Y, models_src2dst_list, models_dst2src_list, optimizer, loss_fn)
                # wandb.log({
                #     "loss_src2dst": losses_src2dst,
                #     "loss_dst2src": losses_dst2src,
                # }) XXX get this wandb logged plz
    return time.time() - time_start

@click.command()
@click.option("--datasets", "-da", type=str, multiple=True)
@click.option("--device", "-d", type=str)
@click.option("--save-path", "-s", type=str)
def main(datasets: List[str], device: str, save_path: str):
    """
    Run with:
    python3 owlergpt/commands/pokemon.py -da arguana -d cuda:3 -s /mnt/align3_drive/adrianoh/dl_final_project_layers_2plus
    """
    assert all(dataset in DATASETS for dataset in datasets)
    save_path = Path(save_path)
    # TODO(Adriano): don't hardcode plz
    embeddings_path = Path("/mnt/align3_drive/adrianoh/dl_final_project_embeddings")
    for dataset in datasets:
        model2embeddings = get_all_embeddings(embeddings_path, dataset, device)
        unordered_pairs_all = [(MODEL_NAMES[i], MODEL_NAMES[j]) for j in range(len(MODEL_NAMES)) for i in range(j)]
        random.seed(55)
        random.shuffle(unordered_pairs_all)
        unordered_pairs_block_size: int = math.ceil(len(unordered_pairs_all) / 3)
        unordered_pairs_blocks = [unordered_pairs_all[i:i+unordered_pairs_block_size] for i in range(0, len(unordered_pairs_all), unordered_pairs_block_size)]
        for idx, unordered_pairs in enumerate(unordered_pairs_blocks):
            print(f"Processing block {idx+1}/{len(unordered_pairs_blocks)}")
            embeddings_tensors_src: List[torch.Tensor] = []
            embeddings_tensors_dst: List[torch.Tensor] = []
            for src, dst in tqdm.tqdm(unordered_pairs):
                # Append by ptr ideally
                embeddings_tensors_src.append(model2embeddings[src])
                embeddings_tensors_dst.append(model2embeddings[dst])
            assert len(embeddings_tensors_src) == len(unordered_pairs)
            assert len(embeddings_tensors_dst) == len(unordered_pairs)

            print("creating models")
            models_src2dst: List[List[MLP]] = []
            models_dst2src: List[List[MLP]] = []
            # NOTE: maybe we just do layers <= len = 7 instead of going all the way to 10 (this is already like 30% of the models' depths)
            n_layers_list = list(range(2,8)) # Maybe two blocks: [2,3], [4,5], [6,7] ???
            # assert len(n_layers_list) <= 3
            for (src, dst), src_emb, dst_emb in tqdm.tqdm(list(zip(unordered_pairs, embeddings_tensors_src, embeddings_tensors_dst))):
                src_dim, dst_dim = src_emb.shape[1], dst_emb.shape[1]
                models_src2dst.append([
                    MLP(
                        input_dim=src_dim, 
                        output_dim=dst_dim, 
                        source_model_name=src, 
                        target_model_name=dst, 
                        dataset=dataset, 
                        hidden_dims=None, # NOTE: default of using higher dims
                        num_layers=n
                    ).to(device) for n in n_layers_list
                ])
                models_dst2src.append([
                    MLP(
                        input_dim=dst_dim, 
                        output_dim=src_dim, 
                        source_model_name=dst, 
                        target_model_name=src, 
                        dataset=dataset,
                        hidden_dims=None, # NOTE: default of using higher dims
                        num_layers=n
                    ).to(device) for n in n_layers_list
                ])
            assert len(models_src2dst) == len(models_dst2src) == len(unordered_pairs)

            print("training for 1 epoch")
            assert isinstance(models_src2dst, list)
            assert isinstance(models_dst2src, list)
            assert all(isinstance(models_src2dst[i], list) for i in range(len(models_src2dst)))
            assert all(isinstance(models_dst2src[i], list) for i in range(len(models_dst2src)))
            assert all(all(isinstance(models_src2dst[i][j], MLP) for j in range(len(models_src2dst[i]))) for i in range(len(models_src2dst)))
            assert all(all(isinstance(models_dst2src[i][j], MLP) for j in range(len(models_dst2src[i]))) for i in range(len(models_dst2src)))
            train_datasets = [EmbeddingDataset(embeddings_src, embeddings_dst) for embeddings_src, embeddings_dst in zip(embeddings_tensors_src, embeddings_tensors_dst)]
            assert len(train_datasets) == len(models_src2dst) == len(models_dst2src)
            num_epochs = 80 # XXX debug
            batch_size = 512 # TODO(Adriano) is this a good hyperparameter?
            loss_fn = nn.MSELoss()
            time_taken = training_run(
                train_datasets,
                models_src2dst,
                models_dst2src,
                loss_fn,
                num_epochs,
                batch_size,
            )
            print(f"Batch size {batch_size}: {time_taken:.2f} seconds")
            time_per_layer_per_epoch = time_taken / num_epochs / len(models_src2dst + models_dst2src)
            print(f"Time per layer per epoch: {time_per_layer_per_epoch:.2f} seconds")
            print("saving models")
            for model_src2dst, model_dst2src in zip(models_src2dst, models_dst2src):
                for model_src2dst_layer, model_dst2src_layer in zip(model_src2dst, model_dst2src):
                    src_model_name, dst_model_name = model_src2dst_layer.source_model_name, model_src2dst_layer.target_model_name
                    src_model_name_path, dst_model_name_path = src_model_name.replace("/", "_"), dst_model_name.replace("/", "_")
                    model_src2dst_layer.save_to_folder_path(
                        save_path / f"{src_model_name_path}_{dst_model_name_path}" / f"{dataset}" / str(model_src2dst_layer.num_layers)
                    )
                    model_dst2src_layer.save_to_folder_path(
                        save_path / f"{dst_model_name_path}_{src_model_name_path}" / f"{dataset}" / str(model_dst2src_layer.num_layers)
                    )
            print("clearing gpu memory")
            del models_src2dst, models_dst2src, train_datasets
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

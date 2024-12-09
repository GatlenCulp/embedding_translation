"""Configuration for a single embedding translation experiment."""

from typing import Literal

import wandb
from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    """Configuration for a single embedding translation experiment."""

    # Dataset config
    dataset_name: str  # e.g. "HotPotQA"
    dataset_split: Literal["train", "test"] = "train"
    dataset_size: int

    # Source/Target embedding models
    source_model: str  # e.g. "T5-small"
    target_model: str  # e.g. "GPT-2"
    source_dim: int
    target_dim: int

    # Architecture config
    architecture: Literal["linear", "mlp_1layer", "mlp_2layer"]
    hidden_dims: list[int]  # e.g. [] for linear, [512] for 1-layer, [512, 256] for 2-layer

    # Training config
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    optimizer: str = "Adam"
    loss_fn: str = "MSE"


def init_wandb_run(
    config: ExperimentConfig,
    project: str = "embedding-translators",
    entity: str | None = None,
    save_code: bool = True,
    group: str | None = None,
) -> wandb.Run:
    """Initialize a W&B run with the given configuration.

    :param config: Experiment configuration
    :param project: W&B project name
    :param entity: W&B entity (username or team name)
    :param save_code: Whether to save code to W&B
    :param group: Optional group name for related experiments
    :return: Initialized W&B run
    """
    # Create a meaningful run name
    run_name = f"{config.source_model}_to_{config.target_model}_{config.architecture}"

    # Initialize run
    run = wandb.init(
        project=project,
        entity=entity,
        config=config.model_dump(),
        name=run_name,
        group=group,
        save_code=save_code,
        tags=[
            config.dataset_name,
            config.architecture,
            f"src_{config.source_model}",
            f"tgt_{config.target_model}",
        ],
    )

    # Define custom charts/tables
    wandb.define_metric("train/mse", summary="min")
    wandb.define_metric("val/mse", summary="min")
    wandb.define_metric("test/mse", summary="min")

    # CKA similarity metrics
    wandb.define_metric("metrics/cka_similarity", summary="max")
    wandb.define_metric("metrics/knn_jaccard", summary="max")

    return run


def log_training_step(
    epoch: int,
    step: int,
    train_loss: float,
    val_loss: float | None = None,
    learning_rate: float | None = None,
    **additional_metrics,
):
    """Log training metrics for a single step.

    :param epoch: Current epoch number
    :param step: Current step within epoch
    :param train_loss: Training MSE loss
    :param val_loss: Optional validation MSE loss
    :param learning_rate: Optional current learning rate
    :param additional_metrics: Any additional metrics to log
    """
    metrics = {
        "train/mse": train_loss,
        "train/epoch": epoch,
        "train/step": step,
    }

    if val_loss is not None:
        metrics["val/mse"] = val_loss

    if learning_rate is not None:
        metrics["train/learning_rate"] = learning_rate

    # Add any additional custom metrics
    for key, value in additional_metrics.items():
        metrics[f"metrics/{key}"] = value

    wandb.log(metrics)


def log_evaluation_metrics(
    mse: float,
    cka_similarity: float,
    knn_jaccard: float,
    embedding_visualizations: wandb.Image | None = None,
    confusion_matrix: wandb.Table | None = None,
):
    """Log final evaluation metrics.

    :param mse: Test set MSE
    :param cka_similarity: CKA similarity score
    :param knn_jaccard: k-NN Jaccard similarity score
    :param embedding_visualizations: Optional UMAP visualization
    :param confusion_matrix: Optional confusion matrix as wandb.Table
    """
    metrics = {
        "test/mse": mse,
        "metrics/cka_similarity": cka_similarity,
        "metrics/knn_jaccard": knn_jaccard,
    }

    if embedding_visualizations is not None:
        metrics["visualizations/umap"] = embedding_visualizations

    if confusion_matrix is not None:
        metrics["visualizations/confusion_matrix"] = confusion_matrix

    wandb.log(metrics)

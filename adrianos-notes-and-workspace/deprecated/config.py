"""Configuration schema for embedding translation experiments."""

from enum import Enum

from pydantic import BaseModel
from pydantic import Field
from pydantic import validator


class ModelType(str, Enum):
    """Types of models available for embedding generation."""

    T5 = "t5"
    GPT = "gpt"
    OTHER = "other"  # Placeholder for additional model


class ModelConfig(BaseModel):
    """Configuration for a single embedding model."""

    name: str
    type: ModelType
    model_id: str
    embedding_dim: int


class TranscoderConfig(BaseModel):
    """Configuration for a transcoder architecture.

    num_layers: Number of non-linear layers (-1 for identity, 0 for linear)
    rank_proportions: Sequence of floats defining relative size of each hidden dimension
                     w.r.t. previous layer (or input for first layer)
    """

    num_layers: int
    rank_proportions: list[float]
    input_dim: int
    output_dim: int

    @validator("rank_proportions")
    def validate_rank_proportions(cls, v, values):
        """Validate that rank proportions match the number of layers and dimensions."""
        if "num_layers" not in values:
            raise ValueError("num_layers must be provided before rank_proportions")

        num_layers = values["num_layers"]
        if num_layers < -1:
            raise ValueError("num_layers must be >= -1")

        if num_layers == -1:  # Identity function
            if v != [1.0]:
                raise ValueError("Identity function must have rank_proportions=[1.0]")
        elif num_layers == 0:  # Linear function
            if v != [1.0]:
                raise ValueError("Linear function must have rank_proportions=[1.0]")
        elif len(v) != num_layers + 1:
            raise ValueError(
                f"MLP with {num_layers} layers must have {num_layers + 1} rank proportions"
            )

        return v

    @validator("output_dim")
    def validate_dimensions(cls, v, values):
        """Validate that output dimension matches input_dim * product of rank_proportions."""
        if all(k in values for k in ["input_dim", "rank_proportions"]):
            expected = values["input_dim"]
            for prop in values["rank_proportions"]:
                expected *= prop
            if abs(expected - v) > 1e-6:  # Using small epsilon for float comparison
                raise ValueError(
                    f"Output dimension {v} does not match expected "
                    f"dimension {expected} (input_dim * product(rank_proportions))"
                )
        return v


class DatasetConfig(BaseModel):
    """Configuration for a dataset."""

    name: str
    train_path: str
    test_path: str


class ExperimentConfig(BaseModel):
    """Root configuration for all experiments."""

    models: list[ModelConfig]
    transcoders: list[TranscoderConfig]
    datasets: list[DatasetConfig]
    k_values: list[int] = Field(default=[1, 5, 10])  # k values for k-NN evaluation
    seed: int = 42
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4

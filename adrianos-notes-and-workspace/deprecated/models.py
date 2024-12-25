"""Transcoder model implementations."""

import torch
from torch import nn


class Transcoder(nn.Module):
    """Unified transcoder for embedding translation.

    Args:
        input_dim: Dimension of input embedding space
        output_dim: Dimension of output embedding space
        num_layers: Number of non-linear layers
            -1 => identity function
            0 => linear function
            >0 => MLP with this many hidden layers
        rank_proportions: Sequence of floats defining relative size of each dimension
            w.r.t. previous layer (or input for first layer)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        rank_proportions: list[float],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.rank_proportions = rank_proportions

        if num_layers < -1:
            raise ValueError("num_layers must be >= -1")

        if num_layers == -1:
            if input_dim != output_dim:
                raise ValueError("Identity function requires input_dim == output_dim")
            self.layers = nn.Identity()
            return

        # For both linear (num_layers=0) and MLP (num_layers>0),
        # we build a sequence of linear layers
        dims = [input_dim]
        for prop in rank_proportions[:-1]:  # All but last proportion
            dims.append(int(dims[-1] * prop))
        dims.append(output_dim)  # Last dimension is output_dim

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Add non-linearity and dropout for hidden layers
            if i < len(dims) - 2:  # Not the last layer
                layers.extend([nn.ReLU(), nn.Dropout(dropout)])

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def create_transcoder(
    input_dim: int,
    output_dim: int,
    num_layers: int,
    rank_proportions: list[float],
    dropout: float = 0.1,
) -> nn.Module:
    """Factory function to create transcoder models."""
    return Transcoder(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        rank_proportions=rank_proportions,
        dropout=dropout,
    )

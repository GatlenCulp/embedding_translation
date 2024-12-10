from __future__ import annotations

import torch
from torch import nn

from owlergpt.modern.schemas import EmbeddingDatasetInformation


################################ STITCHES AVAILABLE ################################


class StitchNNModule(nn.Module):
    """Parent of all stitches."""

    def __init__(
        self,
        embedding_dataset_information_src: EmbeddingDatasetInformation,
        embedding_dataset_information_target: EmbeddingDatasetInformation,
    ):
        raise NotImplementedError("Stitch is an abstract class")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Stitch is an abstract class")


class AffineStitch(StitchNNModule):
    """A linear layer wrapper."""

    def __init__(
        self,
        embedding_dataset_information_src: EmbeddingDatasetInformation,
        embedding_dataset_information_target: EmbeddingDatasetInformation,
    ):
        super().__init__()
        self.source_dim = embedding_dataset_information_src.model_dimension
        self.target_dim = embedding_dataset_information_target.model_dimension
        self.stitch = nn.Linear(self.source_dim, self.target_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stitch(x)

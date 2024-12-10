"""Load embeddings from safe tensor file"""

from pathlib import Path

import numpy as np
import safetensors.numpy
from loguru import logger

from src.utils.general_setup import setup


rng = setup("load_embeddings")

PROJ_ROOT = Path(__file__).parent.parent


def load_embeddings(data_path: Path) -> dict[str, np.ndarray]:
    """Loads the embeddings from a safe tensor file."""
    logger.debug(f"Loading {data_path} as embeddings...")
    return safetensors.numpy.load_file(data_path)

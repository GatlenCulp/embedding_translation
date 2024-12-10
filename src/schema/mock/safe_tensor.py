"""Module for generating mock .safetensor files with randomized data.

Used for testing and development purposes.
"""

from pathlib import Path

import numpy as np
from loguru import logger
from safetensors.numpy import save_file

from src.utils.general_setup import setup


rng = setup("safe_tensor")


def generate_mock_safetensor(
    shape: tuple[int, ...] | list[tuple[int, ...]],
    output_path: str | Path,
    tensor_names: list[str] | None = None,
    dtype: np.dtype = np.float32,
) -> Path:
    """Generate a mock .safetensor file with random data.

    :param shape: Either a single shape tuple or list of shape tuples for multiple tensors
    :param output_path: Path where the .safetensor file will be saved
    :param tensor_names: Optional list of names for the tensors. If None, will use default names
    :param dtype: NumPy dtype for the arrays
    :param seed: Optional random seed for reproducibility
    :return: Path to the generated file
    """
    # Convert to Path object and ensure parent directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert single shape to list for consistent processing
    shapes = [shape] if isinstance(shape, tuple) else shape

    # Generate default tensor names if not provided
    if tensor_names is None:
        tensor_names = [f"tensor_{i}" for i in range(len(shapes))]

    # Ensure we have the same number of names as shapes
    if len(tensor_names) != len(shapes):
        raise ValueError("Number of tensor names must match number of shapes")

    # Generate random arrays and convert to torch tensors
    tensors: dict[str, np.dtype] = {}
    for name, shape in zip(tensor_names, shapes, strict=False):
        array = rng.random(size=shape).astype(dtype)
        tensors[name] = array

    # Save tensors
    save_file(tensors, output_path)

    return output_path


def generate_mock_safetensor_batch(
    shapes: list[tuple[int, ...]],
    output_dir: str | Path,
    num_files: int,
    prefix: str = "mock_tensor",
    dtype: np.dtype = np.float32,
) -> list[Path]:
    """Generate multiple mock .safetensor files.

    :param shapes: List of shapes for tensors in each file
    :param output_dir: Directory where files will be saved
    :param num_files: Number of files to generate
    :param prefix: Prefix for generated filenames
    :param dtype: NumPy dtype for the arrays
    :return: List of paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []
    for i in range(num_files):
        output_path = output_dir / f"{prefix}_{i}.safetensors"
        generated_files.append(
            generate_mock_safetensor(shape=shapes, output_path=output_path, dtype=dtype)
        )

    return generated_files


if __name__ == "__main__":
    # Example usage
    shapes = [(3, 224, 224), (1000,)]  # Example shapes for image and class tensors

    # Generate a single file
    single_file = generate_mock_safetensor(
        shape=shapes,
        output_path="data/embeddings/single_tensor.safetensors",
        tensor_names=["image", "labels"],
    )
    logger.info(f"Generated single file: {single_file}")

    # Generate multiple files
    batch_files = generate_mock_safetensor_batch(
        shapes=shapes, output_dir="data/embeddings", num_files=5, prefix="mock_dataset"
    )
    logger.info(f"Generated {len(batch_files)} files in batch")

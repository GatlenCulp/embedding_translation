"""Reducing the dimensionality of our data using UMAP.

Description: Will take all of our ground truth embeddings and generated embeddings
together and plot them on a single 2D interactive plot.ly graph. Between each of the
embeddings will be an arrow from the original to the translated embedding labeled
with the distance between one and the other.
"""

import numpy as np
import umap
from loguru import logger
from sklearn.preprocessing import StandardScaler

from src.utils.general_setup import setup


setup("reduce_embedding_dims")


# %% Transformation functions
def reduce_embeddings_dimensionality(
    embeddings_dict: dict[str, np.ndarray],
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> dict[str, np.ndarray]:
    """Reduce dimensionality of multiple embedding spaces using UMAP.

    :param dict[str, np.ndarray] embeddings_dict: Dictionary mapping embedding names to arrays
        Each array should have shape (n_samples, n_features) where:
        - n_samples: Number of data points in that embedding space
        - n_features: Dimensionality of the original embedding space (e.g., 768 for BERT)
    :param int random_state: Random seed for reproducibility
    :param int n_neighbors: UMAP parameter for local neighborhood size
    :param float min_dist: UMAP parameter for minimum distance between points
    :return: Dictionary with same keys but values reduced to shape (n_samples, 2)
    :rtype: dict[str, np.ndarray]
    """
    logger.info(f"Starting dimensionality reduction with {len(embeddings_dict)} embedding spaces")

    # Combine all embeddings
    all_embeddings = np.vstack(list(embeddings_dict.values()))
    logger.debug(f"Combined shape: {all_embeddings.shape}")

    # Standardize features
    logger.debug("Standardizing features...")
    scaler = StandardScaler()
    all_embeddings_scaled = scaler.fit_transform(all_embeddings)

    # Reduce dimensionality
    logger.info(f"Applying UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})")
    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    projections = reducer.fit_transform(all_embeddings_scaled)

    # Split projections
    start_idx = 0
    projection_dict = {}
    for name, emb in embeddings_dict.items():
        end_idx = start_idx + len(emb)
        projection_dict[name] = projections[start_idx:end_idx]
        logger.debug(f"Split {name}: {projections[start_idx:end_idx].shape}")
        start_idx = end_idx

    logger.success("Dimensionality reduction completed")
    return projection_dict


def main() -> None:
    """Example usage of embedding dimensionality reduction."""
    # Create sample embeddings (simulating BERT-like 768-dimensional embeddings)
    rng = np.random.default_rng(42)
    n_samples = 100

    sample_embeddings = {
        "original": rng.normal(0, 1, (n_samples, 768)),
        "translated": rng.normal(0.5, 1, (n_samples, 768)),
        "ground_truth": rng.normal(-0.5, 1, (n_samples, 768)),
    }

    logger.info("Created sample embeddings:")
    for name, emb in sample_embeddings.items():
        logger.info(f"  {name}: {emb.shape}")

    # Reduce dimensionality
    reduced_embeddings = reduce_embeddings_dimensionality(
        embeddings_dict=sample_embeddings, random_state=42, n_neighbors=15, min_dist=0.1
    )

    # Print results
    logger.info("Reduced embedding dimensions:")
    for name, emb in reduced_embeddings.items():
        logger.info(f"  {name}: {emb.shape}")


if __name__ == "__main__":
    main()

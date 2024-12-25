"""K-nearest neighbors labeling for semantic search evaluation.

Description: Takes training and test datasets and performs k-nearest neighbors search
to find the closest matches between them. Results are stored in a SemanticSearchEvaluation
model for further analysis of semantic preservation during embedding translation.

NOTE: HAS NOT BEEN CHECKED.
"""

from typing import Literal

import numpy as np
from loguru import logger
from scipy.spatial.distance import cdist

from src.logic.load_embeddings import load_embeddings
from src.schema.SemanticSearch import EmbeddingDatasetInformation
from src.schema.SemanticSearch import SemanticSearchEvaluation
from src.utils.general_setup import setup


setup("knn_labeling")


def find_k_nearest_neighbors(
    training_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    k: int = 1,
    distance_function: Literal["euclidean"] = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """Find k-nearest neighbors for each test embedding in the training set.

    :param np.ndarray training_embeddings: Reference embeddings (n_train_samples, n_features)
    :param np.ndarray test_embeddings: Query embeddings (n_test_samples, n_features)
    :param int k: Number of nearest neighbors to find
    :param Literal["euclidean"] distance_function: Distance metric to use
    :return: Tuple of (indices, distances) arrays
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    logger.debug(
        f"Computing distances between {len(test_embeddings)} test and {len(training_embeddings)} training samples"
    )

    # Compute pairwise distances
    distances = cdist(test_embeddings, training_embeddings, metric=distance_function)

    # Find k smallest distances and their indices
    nearest_indices = np.argpartition(distances, k, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)

    return nearest_indices, nearest_distances


def create_semantic_search_evaluation(
    training_dataset: EmbeddingDatasetInformation,
    test_dataset: EmbeddingDatasetInformation,
    k: int = 1,
    distance_function: Literal["euclidean"] = "euclidean",
) -> SemanticSearchEvaluation:
    """Create a semantic search evaluation from training and test datasets.

    :param dict training_dataset: Dictionary containing embeddings and metadata for training set
    :param dict test_dataset: Dictionary containing embeddings and metadata for test set
    :param int k: Number of nearest neighbors to find
    :param Literal["euclidean"] distance_function: Distance metric to use
    :return: Evaluation results in a SemanticSearchEvaluation model
    :rtype: SemanticSearchEvaluation
    """
    logger.info("Starting semantic search evaluation...")

    # Extract embeddings and record IDs
    training_embeddings = load_embeddings(training_dataset.dataset_filepath)[
        "embeddings"
    ]
    test_embeddings = load_embeddings(test_dataset.dataset_filepath)["embeddings"]

    training_embeddings_size = len(training_embeddings)
    test_embeddings_size = len(test_embeddings)

    training_record_ids = np.arange(training_embeddings_size)
    test_record_ids = np.arange(
        start=training_embeddings_size, stop=test_embeddings_size
    )

    # Create mapping from record_id to label
    training_labels_map = dict(
        zip(training_record_ids, training_record_ids, strict=False)
    )

    # Find nearest neighbors
    indices, distances = find_k_nearest_neighbors(
        training_embeddings=training_embeddings,
        test_embeddings=test_embeddings,
        k=k,
        distance_function=distance_function,
    )

    # Create nearest_neighbors dictionary
    nearest_neighbors = {}
    for i, test_id in enumerate(test_record_ids):
        neighbors = {}
        for j, neighbor_idx in enumerate(indices[i]):
            neighbor_id = training_record_ids[neighbor_idx]
            neighbors[neighbor_id] = float(distances[i][j])
        nearest_neighbors[test_id] = neighbors

    # Create labels dictionary (using first nearest neighbor's label)
    labels = {}
    for i, test_id in enumerate(test_record_ids):
        nearest_neighbor_idx = indices[i][0]
        nearest_neighbor_id = training_record_ids[nearest_neighbor_idx]
        labels[test_id] = training_labels_map[nearest_neighbor_id]

    logger.success("Semantic search evaluation completed")

    return SemanticSearchEvaluation(
        training_dataset=training_dataset,
        test_dataset=test_dataset,
        distance_function=distance_function,
        k=k,
        nearest_neighbors=nearest_neighbors,
        labels=labels,
    )


def main() -> None:
    """Example usage of semantic search evaluation."""
    # Create sample datasets
    rng = np.random.default_rng(42)
    n_samples = 100
    n_features = 768

    # Sample dataset info
    sample_ingestion_settings = {"batch_size": 32, "max_length": 512, "device": "cpu"}

    # Sample training dataset
    training_dataset = {
        "embeddings": rng.normal(0, 1, (n_samples, n_features)),
        "record_ids": [f"train_{i}" for i in range(n_samples)],
        "labels": [f"label_{i%5}" for i in range(n_samples)],
        "dataset_info": {
            "name": "sample_training",
            "embedding_model_name": "bert-base-uncased",
            "embedding_model_type": "openai",
            "embedding_dimension": n_features,
            "text_dataset_name": "sample_text_dataset",
            "chromadb_collection_name": "sample_collection",
            "ingestion_settings": sample_ingestion_settings,
            "dataset_filepath": "path/to/training/data",
        },
    }

    # Sample test dataset
    test_dataset = {
        "embeddings": rng.normal(0.5, 1, (n_samples // 2, n_features)),
        "record_ids": [f"test_{i}" for i in range(n_samples // 2)],
        "dataset_info": {
            "name": "sample_test",
            "embedding_model_name": "bert-base-uncased",
            "embedding_model_type": "openai",
            "embedding_dimension": n_features,
            "text_dataset_name": "sample_text_dataset",
            "chromadb_collection_name": "sample_collection",
            "ingestion_settings": sample_ingestion_settings,
            "dataset_filepath": "path/to/test/data",
        },
    }

    # Run evaluation
    evaluation = create_semantic_search_evaluation(
        training_dataset=training_dataset,
        test_dataset=test_dataset,
        k=3,
    )

    # Print sample results
    logger.info(f"Evaluated {len(evaluation.labels)} test samples")
    sample_test_id = next(iter(evaluation.labels.keys()))
    logger.info(f"Sample test ID: {sample_test_id}")
    logger.info(f"  Label: {evaluation.labels[sample_test_id]}")
    logger.info(f"  Nearest neighbors: {evaluation.nearest_neighbors[sample_test_id]}")


if __name__ == "__main__":
    main()

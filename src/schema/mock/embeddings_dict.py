import numpy as np


def create_example_embeddings(
    n_samples: int = 100,
    n_dimensions: int = 768,  # Common embedding dimension for models like BERT
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Create example embedding spaces for visualization.

    :param int n_samples: Number of samples in each embedding space
    :param int n_dimensions: Number of dimensions for each embedding
    :param int random_state: Random seed for reproducibility
    :return: Dictionary of embedding spaces
    :rtype: dict[str, np.ndarray]
    """
    rng = np.random.default_rng(random_state)

    # Create three different embedding spaces
    embeddings_dict = {
        "Original": rng.standard_normal((n_samples, n_dimensions)),
        "Translated": rng.standard_normal((n_samples, n_dimensions)) + 2,
        "Scaled": rng.standard_normal((n_samples, n_dimensions)) * 2,
    }

    return embeddings_dict

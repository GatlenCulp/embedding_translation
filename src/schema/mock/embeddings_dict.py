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
    gen = np.random.Generator(random_state)

    # Create three different embedding spaces
    embeddings_dict = {
        "Original": gen.normal(0, 1, (n_samples, n_dimensions)),
        "Translated": gen.normal(2, 1, (n_samples, n_dimensions)),  # Shifted mean
        "Scaled": gen.normal(0, 2, (n_samples, n_dimensions)),  # Different variance
    }

    return embeddings_dict

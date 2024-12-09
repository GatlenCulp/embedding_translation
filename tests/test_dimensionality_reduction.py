"""Tests for the dimensionality reduction visualization pipeline."""

import numpy as np
import plotly.express as px
import pytest
from plotly.graph_objects import Figure

from src.logic.reduce_embedding_dims import reduce_embeddings_dimensionality
from src.viz.dimensionality_reduction import visualize_embeddings


@pytest.fixture
def rng() -> np.random.Generator:
    """Create a random number generator with a fixed seed.

    :return: NumPy random number generator
    :rtype: np.random.Generator
    """
    return np.random.default_rng(42)


@pytest.fixture
def iris_embeddings() -> dict[str, np.ndarray]:
    """Create embeddings dictionary from iris dataset.

    :return: Dictionary mapping species to their feature arrays
    :rtype: dict[str, np.ndarray]
    """
    iris_df = px.data.iris()
    features = iris_df.loc[:, :"petal_width"]

    return {
        species: features[iris_df.species == species].to_numpy()
        for species in iris_df.species.unique()
    }


def test_visualization_config(rng: np.random.Generator) -> None:
    """Test visualization configuration options.

    :param np.random.Generator rng: Random number generator
    """
    # Create simple 2D embeddings
    embeddings = {
        "A": rng.random((10, 2)),
        "B": rng.random((10, 2)),
    }

    # Test custom dimensions
    config = {"width": 1000, "height": 800}
    fig = visualize_embeddings(embeddings, config=config)

    assert fig.layout.width == 1000, "Should use custom width"
    assert fig.layout.height == 800, "Should use custom height"


def test_empty_embeddings() -> None:
    """Test handling of empty embeddings dictionary."""
    with pytest.raises(ValueError):
        visualize_embeddings({})


def test_invalid_dimensions(rng: np.random.Generator) -> None:
    """Test handling of invalid input dimensions.

    :param np.random.Generator rng: Random number generator
    """
    # Test 1D input
    invalid_embeddings = {
        "A": rng.random(10),
    }
    with pytest.raises(ValueError):
        visualize_embeddings(invalid_embeddings)

    # Test 3D input
    invalid_embeddings = {
        "A": rng.random((10, 3)),
    }
    with pytest.raises(ValueError):
        visualize_embeddings(invalid_embeddings)

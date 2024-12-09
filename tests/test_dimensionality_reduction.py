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


def test_visualization_pipeline(iris_embeddings: dict[str, np.ndarray]) -> None:
    """Test the full visualization pipeline using iris dataset.

    :param dict[str, np.ndarray] iris_embeddings: Fixture containing iris embeddings
    """
    # Verify input data
    assert len(iris_embeddings) == 3, "Should have 3 species"
    for embeddings in iris_embeddings.values():
        assert embeddings.shape[1] == 4, "Should be 4D features"

    # Reduce dimensionality
    reduced_embeddings = reduce_embeddings_dimensionality(iris_embeddings)

    # Verify reduction
    assert len(reduced_embeddings) == len(iris_embeddings), "Should preserve number of species"
    for embeddings in reduced_embeddings.values():
        assert embeddings.shape[1] == 2, "Should reduce to 2D"
        assert not np.any(np.isnan(embeddings)), "Should not contain NaN values"

    # Create visualization
    fig = visualize_embeddings(reduced_embeddings)

    # Verify figure
    assert isinstance(fig, Figure), "Should return a plotly Figure"
    assert len(fig.data) == 3, "Should have one trace per species"
    assert fig.layout.title.text == "Embedding Space Comparison", "Should have correct title"


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

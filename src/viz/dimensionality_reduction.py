"""Reducing the dimensionality of our data using UMAP.

Description: Will take all of our ground truth embeddings and generated embeddings
together and plot them on a single 2D interactive plot.ly graph. Between each of the
embeddings will be an arrow from the original to the translated embedding labeled
with the distance between one and the other.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger

from src.logic.reduce_embedding_dims import reduce_embeddings_dimensionality
from src.utils.general_setup import setup
from src.viz.save_figure import save_figure


setup("dimensionality_reduction")


# %% Visualization function
def _plot_embedding_spaces(
    projection_dict: dict[str, np.ndarray],
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """Create an interactive plot of multiple 2D embedding spaces.

    :param dict[str, np.ndarray] projection_dict: Dictionary of 2D projections
    :param int width: Width of the plot in pixels
    :param int height: Height of the plot in pixels
    :return: Plotly figure with embedding visualization
    :rtype: go.Figure
    """
    logger.info("Creating visualization...")

    fig = go.Figure()
    colors = px.colors.qualitative.Set3

    for i, (name, proj) in enumerate(projection_dict.items()):
        logger.debug(f"Adding trace for {name}")
        fig.add_trace(
            go.Scatter(
                x=proj[:, 0],
                y=proj[:, 1],
                mode="markers",
                name=name,
                marker=dict(color=colors[i % len(colors)]),
            )
        )

    # Update layout
    fig.update_layout(
        title="Embedding Space Comparison",
        showlegend=True,
        width=width,
        height=height,
    )

    fig.update_xaxes(showticklabels=True, title=None)
    fig.update_yaxes(showticklabels=True, title=None)

    logger.success("Visualization created")
    return fig


# %% Combined function for backwards compatibility
def _visualize_embedding_translations(
    embeddings_dict: dict[str, np.ndarray],
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> go.Figure:
    """Create a 2D visualization of multiple embedding spaces with transition arrows.

    :param dict[str, np.ndarray] embeddings_dict: Dictionary mapping embedding names to arrays
    :param int random_state: Random seed for reproducibility
    :param int n_neighbors: UMAP parameter for local neighborhood size
    :param float min_dist: UMAP parameter for minimum distance between points
    :return: Plotly figure with embedding visualization
    :rtype: go.Figure
    """
    logger.info("Starting combined visualization process")

    projections = reduce_embeddings_dimensionality(
        embeddings_dict,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    return _plot_embedding_spaces(projections)


def visualize_embeddings(
    embeddings_dict: dict[str, np.ndarray],
    config: dict = None,
) -> tuple[go.Figure, go.Figure]:
    """Create visualizations for arbitrary embedding spaces.

    :param dict[str, np.ndarray] embeddings_dict: Dictionary mapping embedding names to arrays
    :param dict config: Configuration for visualization parameters (optional)
    :return: Tuple of (combined method figure, separate method figure)
    :rtype: tuple[go.Figure, go.Figure]
    """
    # Set default config if none provided
    if config is None:
        config = {
            "random_state": 42,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "width": 800,
            "height": 600,
        }

    logger.info("Creating visualizations...")

    # Method 1: Combined
    fig1 = _visualize_embedding_translations(
        embeddings_dict,
        random_state=config.get("random_state", 42),
        n_neighbors=config.get("n_neighbors", 15),
        min_dist=config.get("min_dist", 0.1),
    )

    # Method 2: Separate
    projections = reduce_embeddings_dimensionality(
        embeddings_dict,
        random_state=config.get("random_state", 42),
        n_neighbors=config.get("n_neighbors", 15),
        min_dist=config.get("min_dist", 0.1),
    )
    fig2 = _plot_embedding_spaces(
        projections,
        width=config.get("width", 800),
        height=config.get("height", 600),
    )

    return fig1, fig2


def _iris_example() -> None:
    """Run example visualization using iris dataset."""
    logger.info("Starting example with iris dataset")

    # Load iris data
    iris_df = px.data.iris()
    features = iris_df.loc[:, :"petal_width"]

    # Create mock embedding spaces
    embeddings_dict = {
        species: features[iris_df.species == species].to_numpy()
        for species in iris_df.species.unique()
    }
    logger.info(f"Created embeddings for {len(embeddings_dict)} species")

    # Create visualizations
    fig1, fig2 = visualize_embeddings(embeddings_dict)

    # Display and save figures
    fig1.show()
    fig2.show()
    save_figure(fig1, "iris_embeddings_combined")
    save_figure(fig2, "iris_embeddings_separate")

    logger.success("Example completed successfully")


def main() -> None:
    """Run the iris example."""
    _iris_example()


# %% Example usage
if __name__ == "__main__":
    main()

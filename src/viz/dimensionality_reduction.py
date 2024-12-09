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

from src.transforms.reduce_embedding_dims import reduce_embeddings_dimensionality
from src.utils.general_setup import setup


setup("dimensionality_reduction")


# %% Visualization function
def plot_embedding_spaces(
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
def visualize_embedding_translations(
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
    return plot_embedding_spaces(projections)


def main() -> None:
    """Runs example translation using iris dataset."""
    logger.info("Starting example with iris dataset")

    # Load iris data
    iris_df = px.data.iris()
    features = iris_df.loc[:, :"petal_width"]

    # Create mock embedding spaces
    embeddings_dict = {
        species: features[iris_df.species == species].values for species in iris_df.species.unique()
    }
    logger.info(f"Created embeddings for {len(embeddings_dict)} species")

    # Method 1: Combined
    logger.info("Testing combined method...")
    fig1 = visualize_embedding_translations(embeddings_dict)
    fig1.show()

    # Method 2: Separate
    logger.info("Testing separate methods...")
    projections = reduce_embeddings_dimensionality(embeddings_dict)
    fig2 = plot_embedding_spaces(projections)
    fig2.show()

    logger.success("Example completed successfully")


# %% Example usage
if __name__ == "__main__":
    main()

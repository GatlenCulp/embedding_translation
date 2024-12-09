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


def visualize_embeddings(
    embeddings_dict: dict[str, np.ndarray],
    config: dict | None = None,
) -> go.Figure:
    """Create visualization for arbitrary embedding spaces.

    :param dict[str, np.ndarray] embeddings_dict: Dictionary of 2D embeddings
    :param dict config: Configuration for visualization parameters (optional)
    :return: Plotly figure with embedding visualization
    :rtype: go.Figure
    """
    # Set default config if none provided
    if config is None:
        config = {
            "width": 800,
            "height": 600,
        }

    logger.info("Creating visualization...")
    return _plot_embedding_spaces(
        embeddings_dict,
        width=config.get("width", 800),
        height=config.get("height", 600),
    )

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

    # Create visualization
    fig = visualize_embeddings(embeddings_dict)

    # Display and save figure
    fig.show()
    save_figure(fig, "iris_embeddings")

    logger.success("Example completed successfully")


def main() -> None:
    """Run the iris example."""
    _iris_example()


# %% Example usage
if __name__ == "__main__":
    main()

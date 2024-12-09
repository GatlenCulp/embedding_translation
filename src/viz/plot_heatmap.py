import numpy as np
import plotly.graph_objects as go
from loguru import logger

from src.utils.general_setup import setup
from src.viz.save_figure import save_figure


setup("plot_heatmap")


def _plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    title: str = "Heatmap",
    width: int = 800,
    height: int = 600,
    color_scale: str = "Viridis",
    show_values: bool = True,
    value_format: str = ".2f",
) -> go.Figure:
    """Create a Plotly heatmap from a 2D array.

    :param matrix: 2D numpy array of numerical values
    :param row_labels: Optional list of row labels
    :param col_labels: Optional list of column labels
    :param title: Plot title
    :param width: Figure width in pixels
    :param height: Figure height in pixels
    :param color_scale: Colorscale for the heatmap (e.g. 'Viridis', 'Cividis')
    :param show_values: Whether to overlay cell values on the heatmap
    :param value_format: Format string for the displayed values
    :return: Plotly figure object
    """
    logger.info("Creating heatmap visualization...")

    # Create text annotations for each cell if requested
    text_vals = None
    text_template = None
    if show_values:
        text_vals = np.vectorize(lambda x: f"{x:{value_format}}")(matrix)
        text_template = "%{text}"

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=col_labels,
            y=row_labels,
            colorscale=color_scale,
            text=text_vals,
            texttemplate=text_template,
            textfont={"color": "black"},
            reversescale=False,
            showscale=True,
        )
    )

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis_nticks=len(col_labels) if col_labels else None,
        yaxis_nticks=len(row_labels) if row_labels else None,
    )

    logger.success("Heatmap created successfully")
    return fig


def visualize_heatmap(
    matrix: np.ndarray,
    config: dict | None = None,
) -> go.Figure:
    """Create a heatmap visualization using a configuration dictionary.

    :param matrix: 2D numpy array of values
    :param config: Optional dictionary to specify plot parameters
    :return: Plotly figure object
    """
    if config is None:
        config = {
            "title": "Heatmap",
            "width": 800,
            "height": 600,
            "color_scale": "Viridis",
            "show_values": True,
            "value_format": ".2f",
            "row_labels": None,
            "col_labels": None,
        }

    logger.info("Starting heatmap visualization process")

    fig = _plot_heatmap(
        matrix=matrix,
        row_labels=config.get("row_labels"),
        col_labels=config.get("col_labels"),
        title=config.get("title", "Heatmap"),
        width=config.get("width", 800),
        height=config.get("height", 600),
        color_scale=config.get("color_scale", "Viridis"),
        show_values=config.get("show_values", True),
        value_format=config.get("value_format", ".2f"),
    )

    return fig


def main() -> None:
    logger.info("Running example with random matrix")

    # Example data: A random 10x10 matrix
    matrix = np.random.rand(10, 10)
    row_labels = [f"Row {i}" for i in range(1, 11)]
    col_labels = [f"Col {j}" for j in range(1, 11)]

    config = {
        "title": "Example Heatmap",
        "width": 800,
        "height": 600,
        "color_scale": "Viridis",
        "show_values": True,
        "value_format": ".2f",
        "row_labels": row_labels,
        "col_labels": col_labels,
    }

    fig = visualize_heatmap(matrix, config=config)
    fig.show()
    logger.success("Example heatmap displayed successfully")


if __name__ == "__main__":
    main()

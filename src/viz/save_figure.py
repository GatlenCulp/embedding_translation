"""Save plotly figures as both HTML and PNG files.

Description: Provides functionality to save plotly figures in multiple formats with
configurable settings. Uses CDN for plotlyjs in HTML to reduce file size and
kaleido for high-quality PNG exports.
"""

from pathlib import Path

import plotly.graph_objects as go
from loguru import logger
from plotly.basedatatypes import BaseFigure

from src.utils.general_setup import setup


setup("save_figure")


# %% Save functions
def _save_figure_html(
    fig: BaseFigure,
    filepath: Path,
) -> Path:
    """Save a plotly figure as an HTML file.

    :param BaseFigure fig: The plotly figure to save
    :param Path filepath: Full path including filename and .html extension
    :return: Path to the saved HTML file
    :rtype: Path
    """
    logger.info(f"Saving HTML figure to {filepath}")

    try:
        fig.write_html(
            filepath,
            # include_plotlyjs="cdn",  # Use CDN to reduce file size
            full_html=True,
        )
        logger.success(f"Successfully saved HTML to {filepath}")

    except Exception as e:
        logger.error(f"Failed to save HTML: {e!s}")
        raise

    else:
        return filepath


def _save_figure_png(
    fig: BaseFigure,
    filepath: Path,
    width: int = 1200,
    height: int = 800,
    scale: float = 2.0,
) -> Path:
    """Save a plotly figure as a PNG file.

    :param BaseFigure fig: The plotly figure to save
    :param Path filepath: Full path including filename and .png extension
    :param int width: Width of the figure in pixels
    :param int height: Height of the figure in pixels
    :param float scale: Scale factor for PNG resolution
    :return: Path to the saved PNG file
    :rtype: Path
    """
    logger.info(f"Saving PNG figure to {filepath}")

    try:
        fig.write_image(
            filepath,
            width=width,
            height=height,
            scale=scale,
        )
        logger.success(f"Successfully saved PNG to {filepath}")

    except Exception as e:
        logger.error(f"Failed to save PNG: {e!s}")
        raise

    else:
        return filepath


# %% Combined function
def save_figure(
    fig: BaseFigure,
    filename: str | Path,
    output_dir: str | Path = "data/figs",
    width: int = 1200,
    height: int = 800,
    scale: float = 2.0,
) -> tuple[Path, Path]:
    """Save a plotly figure as both HTML and PNG files.

    :param BaseFigure fig: The plotly figure to save
    :param str | Path filename: Name of the file without extension
    :param str | Path output_dir: Directory to save the files in
    :param int width: Width of the figure in pixels
    :param int height: Height of the figure in pixels
    :param float scale: Scale factor for PNG resolution
    :return: Tuple of paths to the saved HTML and PNG files
    :rtype: tuple[Path, Path]
    """
    # Convert paths to Path objects
    output_dir = Path(output_dir)
    filename = Path(filename).stem  # Get filename without extension

    # Generate file paths
    html_dir = output_dir / "html"
    png_dir = output_dir / "imgs"

    html_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    html_path = html_dir / f"{filename}.html"
    png_path = png_dir / f"{filename}.png"

    # Save both formats
    html_path = _save_figure_html(fig, html_path)
    png_path = _save_figure_png(fig, png_path, width, height, scale)

    return html_path, png_path


def main() -> None:
    """Run example figure saving."""
    logger.info("Starting example figure saving")

    # Create example figure
    fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])], layout={"title": "Example Plot"})

    # Method 1: Combined
    logger.info("Testing combined method...")
    html_path, png_path = save_figure(
        fig=fig, filename="example_plot_combined"
    )


# %% Example usage
if __name__ == "__main__":
    main()

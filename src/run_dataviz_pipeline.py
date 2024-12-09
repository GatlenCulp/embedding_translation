"""Module to run the datavisualization on the results of the main process.

## Experimental Setup

For each combination of:

- Dataset
- Embedding model pair (A, B)
- Direction (A→B and B→A)
- Architecture type and width
  Total: ~36 experiments

## Analysis Pipeline

For each experiment:

1. Train using MSE loss
2. Generate embeddings on test dataset
3. Create UMAP visualization with prompts/labels
4. Generate tables comparing:
   - MSE across model pairs and ranks
   - k-NN edit distance metrics

"""

from pathlib import Path

import numpy as np

from src.logic.anal_dump import anal_dump
from src.schema.mock.embeddings_dict import create_example_embeddings
from src.schema.mock.stitch_summary import create_example_stitch_summary
from src.schema.training_schemas import StitchSummary
from src.utils.general_setup import setup
from src.viz.dimensionality_reduction import visualize_embeddings
from src.viz.plot_heatmap import visualize_heatmap
from src.viz.save_figure import save_figure


setup("run_dataviz_pipeline")

PROJ_ROOT = Path(__file__).parent.parent
rng = np.random.default_rng()


def _load_data_as_stitch_summary(data_path: Path) -> StitchSummary:
    return create_example_stitch_summary()


def dataviz_pipeline(data_path: Path) -> None:
    """Runs entire data visualization pipeline on saved data."""

    stitch_summary = _load_data_as_stitch_summary(data_path=data_path)
    anal_dump(stitch_summary, "test_visualize_embeddings")
    fig = visualize_embeddings(embeddings_dict=create_example_embeddings())
    save_figure(fig, "test_visualize_embeddings")

    fig = visualize_heatmap(matrix=rng.random((10, 10)))
    save_figure(fig, "test_heatmap")


def main() -> None:
    """Runs dataviz pipeline with default config."""
    data_path = PROJ_ROOT / "data" / "data.json"
    dataviz_pipeline(data_path)


if __name__ == "__main__":
    main()

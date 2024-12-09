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

from src.viz.dimensionality_reduction import visualize_embeddings


def dataviz_pipeline(data_path: Path) -> None:
    """Runs entire data visualization pipeline on saved data."""
    embeddings_dict = {}
    visualize_embeddings(embeddings_dict=embeddings_dict)

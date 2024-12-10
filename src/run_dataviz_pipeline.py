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

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import safetensors.numpy
from loguru import logger

from src.logic.anal_dump import DataFile
from src.logic.anal_dump import anal_dump
from src.logic.reduce_embedding_dims import reduce_embeddings_dimensionality
from src.schema.training_schemas import StitchSummary
from src.utils.general_setup import setup
from src.viz.dimensionality_reduction import visualize_embeddings
from src.viz.plot_heatmap import visualize_heatmap
from src.viz.save_figure import save_figure


rng = setup("run_dataviz_pipeline")

PROJ_ROOT = Path(__file__).parent.parent


class DataVizPipeline:
    """Class to run the datavisualization on the results of the main process."""

    @staticmethod
    def _load_data_as_stitch_summary(data_path: Path) -> StitchSummary:
        """Load JSON data from path and convert to StitchSummary."""
        logger.debug(f"Loading {data_path} as summary...")
        with data_path.open() as f:
            data = f.read()
        data = DataFile.model_validate_json(data)
        return StitchSummary.model_validate(data.data)

    @staticmethod
    def _load_embeddings(data_path: Path) -> dict[str, np.ndarray]:
        """Loads the embeddings from a safe tensor file."""
        logger.debug(f"Loading {data_path} as embeddings...")
        return safetensors.numpy.load_file(data_path)

    @staticmethod
    def _run_embedding_viz(stitch_summary: StitchSummary) -> None:
        logger.debug(f"Running _run_embedding_viz with {stitch_summary.display_name}")

        # Get embeddings
        test_stitch_embeddings_dataset = stitch_summary.test_stitch_embeddings
        test_target_embeddings_dataset = stitch_summary.test_experiment_config.target

        # Load embeddings
        test_stitch_embeddings = DataVizPipeline._load_embeddings(
            data_path=test_stitch_embeddings_dataset.dataset_filepath
        )["embeddings"]
        test_target_embeddings = DataVizPipeline._load_embeddings(
            data_path=test_target_embeddings_dataset.dataset_filepath
        )["embeddings"]

        embeddings_dict = {"stitched": test_stitch_embeddings, "target": test_target_embeddings}

        # Reduce dimensionality
        reduced_embeddings = reduce_embeddings_dimensionality(embeddings_dict)

        fig = visualize_embeddings(reduced_embeddings)

        save_figure(fig, f"embedding_viz_{stitch_summary.slug}")

    @staticmethod
    def _get_matrix_from_stich_summary(
        stitch_summaries: list[StitchSummary], stitch_fn: Callable[[StitchSummary], np.ndarray]
    ) -> tuple[list[list[Any]], list[str], list[str]]:
        """Matches stitch_summaries into a matrix, then calls stitch_fn on each.

        Args:
            stitch_summaries: List of stitch summaries to analyze
            stitch_fn: Function to call on each stitch_summary to generate matrix entry

        Returns:
            tuple containing:
            - list[list[Any]]: Matrix of results where rows=source models, cols=target models
            - list[str]: Row labels (source model names)
            - list[str]: Column labels (target model names)
        """
        # Get unique source and target models
        source_models = sorted({s.source_embedding_model_name for s in stitch_summaries})
        target_models = sorted({s.target_embedding_model_name for s in stitch_summaries})

        # Initialize matrix with None values
        matrix = [[None for _ in target_models] for _ in source_models]

        # Create lookup dictionaries for indices
        source_to_idx = {model: idx for idx, model in enumerate(source_models)}
        target_to_idx = {model: idx for idx, model in enumerate(target_models)}

        # Fill matrix
        for stitch in stitch_summaries:
            i = source_to_idx[stitch.source_embedding_model_name]
            j = target_to_idx[stitch.target_embedding_model_name]
            matrix[i][j] = stitch_fn(stitch)

        return matrix, source_models, target_models

    @staticmethod
    def _run_loss_viz(stitch_summary: StitchSummary) -> None:
        logger.debug(f"Running _run_loss_viz with {stitch_summary.display_name}")
        matrix = rng.random((10, 10))
        fig = visualize_heatmap(matrix=matrix)
        save_figure(fig, f"loss_viz_{stitch_summary.slug}")

    @staticmethod
    def _run_isolated_eval(stitch_summary: StitchSummary) -> None:
        anal_dump(stitch_summary, "test_visualize_embeddings")

        DataVizPipeline._run_embedding_viz(stitch_summary)

    @staticmethod
    def run(paths_to_stitch_summaries: list[Path]) -> None:
        """Runs entire data visualization pipeline on saved data."""
        logger.info(f"Running DataViz pipeline with {len(paths_to_stitch_summaries)}")
        stitch_summaries: list[StitchSummary] = [
            DataVizPipeline._load_data_as_stitch_summary(path) for path in paths_to_stitch_summaries
        ]
        for stitch_summary in stitch_summaries:
            # DataVizPipeline._run_isolated_eval(stitch_summary)
            pass
        matrix, row_labels, col_labels = DataVizPipeline._get_matrix_from_stich_summary(
            stitch_summaries,
            lambda stitch: stitch,  # Just pass the stitch summary itself
        )


def main() -> None:
    """Runs dataviz pipeline with default config."""
    stitch_summaries_dir = PROJ_ROOT / "data" / "stitch_summaries"
    data_paths = list(stitch_summaries_dir.glob("*.json"))
    DataVizPipeline.run(data_paths)


if __name__ == "__main__":
    main()

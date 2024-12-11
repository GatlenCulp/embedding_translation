"""Creates StitchSummaries from a given directory which is the format that Adriano saved the stuff in."""

# %%
import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from src.collection_utils import model2model_dimension
from src.DataVizPipeline import save_figure
from src.DataVizPipeline import visualize_heatmap
from src.DataVizPipeline import DataVizPipeline, DataFile
from src.schema.training_schemas import EmbeddingDatasetInformation
from src.schema.training_schemas import ExperimentConfig
from src.schema.training_schemas import StitchSummary
from src.utils.general_setup import setup
from src.logic.anal_dump import anal_dump


setup("StitchSummaryGenerator")

PROJ_ROOT = Path(__file__).parent.parent


class TrainingLogs(BaseModel):
    logs: list[dict[Any, Any]]


class ModelGenerator:
    """Creates StitchSummaries from a given directory which is the format that Adriano saved the stuff in."""

    @staticmethod
    def native_embedding_dataset_info_from_dir(
        native_embeddings_dir: Path,
    ) -> list[EmbeddingDatasetInformation]:
        """Create StitchSummaries from a directory of embeddings.

        Directory structure is:
        dataset_name/embedding_model_name/embedding_model_architecture/embeddings.safetensors

        Args:
            embeddings_dir: Root directory containing embedding folders

        Returns:
            List of StitchSummary objects, one for each embedding
        """
        if not native_embeddings_dir.exists():
            raise ValueError("Expected embeddings_dir to exist")

        embedding_datasets = []

        # Iterate through dataset directories (e.g. fiqa, arguana)
        for dataset_dir in native_embeddings_dir.iterdir():
            if dataset_dir.name.startswith("."):  # Skip hidden files
                continue

            # Iterate through model name directories (e.g. WhereIsAI, BAAI)
            for model_dir in dataset_dir.iterdir():
                for arch_dir in model_dir.iterdir():
                    embeddings_file = arch_dir / "embeddings.safetensors"
                    if not embeddings_file.exists():
                        continue

                    dataset, _, model = dataset_dir.name, model_dir.name, arch_dir.name

                    embedding_dataset = EmbeddingDatasetInformation(
                        embedding_model_name=model,
                        embedding_model_type="huggingface"
                        if "huggingface" in native_embeddings_dir.name
                        else "openai",
                        embedding_dimension=model2model_dimension(model),
                        text_dataset_name=dataset,
                        text_dataset_source="huggingface"
                        if "huggingface" in native_embeddings_dir.name
                        else "openai",
                        dataset_filepath=embeddings_file,
                    )

                    embedding_datasets.append(embedding_dataset)

        return embedding_datasets

    @staticmethod
    def stitches_from_dir_i_guess(dir: Path) -> None:
        """Runs dataviz pipeline with default config."""
        stitch_summaries_dir = PROJ_ROOT / "data" / "stitch_summaries"
        # data_paths = list(stitch_summaries_dir.glob("*.json"))

    @staticmethod
    def get_train_logs(dir: Path) -> list[dict]:
        """Extract training information from stitch directories.

        Args:
            dir: Root directory containing stitch folders, each with info and log files

        Returns:
            List of dictionaries containing info and final log entries for each stitch
        """
        if not dir.exists():
            raise ValueError(f"Directory {dir} does not exist")

        train_values = []

        for stitch_dir in dir.iterdir():
            if not stitch_dir.is_dir():
                continue
            for dataset_dir in stitch_dir.iterdir():
                info_path = dataset_dir / "stitch_info_pairs.json"
                log_path = dataset_dir / "log.jsonl"

                # Skip if either file is missing
                if not info_path.exists() or not log_path.exists():
                    logger.warning(f"Skipping {dataset_dir.name} - missing required files")
                    continue

                try:
                    # Read the info file
                    with info_path.open() as f:
                        info = json.loads(f.read())

                    # Read the last line of the log file
                    with log_path.open() as f:
                        # Skip to the end and read last line
                        log = [json.loads(line) for line in f.readlines()]

                    train_value = {
                        "info": info,
                        "log": log,
                        "dir": str(dataset_dir),  # Include directory for reference
                    }
                    train_values.append(train_value)

                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in {stitch_dir.name}")
                except Exception as e:
                    logger.error(f"Error processing {stitch_dir.name}: {e!s}")

        return train_values

    @staticmethod
    def get_mapping_from_train_values(train_values: list[dict[str, Any]]):
        # Group by source and target, allowing multiple targets per source
        matrix: dict[str, dict[str, Any]] = {}
        for train_value in train_values:
            source = train_value["info"]["source"]
            target = train_value["info"]["target"]
            if source not in matrix:
                matrix[source] = {}
            matrix[source][target] = train_value
        return matrix

    @staticmethod
    def get_mse_matrix_from_matrix(matrix: dict[str, dict[str, Any]], labels: list[str]):
        mse_matrix = [[None for _ in labels] for _ in labels]  # Initialize with zeros
        for i, row_label in enumerate(labels):
            for j, col_label in enumerate(labels):
                if i == j:
                    continue
                try:
                    mse_matrix[i][j] = matrix[row_label][col_label]["log"][-1]["test_mse"]
                except (KeyError, IndexError):
                    continue
        return mse_matrix

    @staticmethod
    def get_mae_matrix_from_matrix(matrix: dict[str, dict[str, Any]], labels: list[str]):
        mae_matrix = [[None for _ in labels] for _ in labels]  # Initialize with zeros
        for i, row_label in enumerate(labels):
            for j, col_label in enumerate(labels):
                if i == j:
                    continue
                try:
                    mae_matrix[i][j] = matrix[row_label][col_label]["log"][-1]["test_mse"]
                except (KeyError, IndexError):
                    continue
        return mae_matrix

    @staticmethod
    def load_data_as_train_logs(data_path: Path) -> StitchSummary:
        """Load JSON data from path and convert to StitchSummary."""
        logger.debug(f"Loading {data_path} as train logs...")
        with data_path.open() as f:
            data = f.read()
        data = DataFile.model_validate_json(data)
        return TrainingLogs.model_validate(data.data)

    @staticmethod
    def stitches_from_directory(
        native_embeddings_dir: Path, stitched_embeddings_dir: Path
    ) -> list[StitchSummary]:
        """Create StitchSummaries from a directory of embeddings.

        Directory structure is:
        dataset_name/embedding_model_name/embedding_model_architecture/embeddings.safetensors

        Args:
            embeddings_dir: Root directory containing embedding folders

        Returns:
            List of StitchSummary objects, one for each embedding
        """
        naitve_embeddings = ModelGenerator.native_embedding_dataset_info_from_dir(
            native_embeddings_dir
        )

        if not stitched_embeddings_dir.exists():
            raise ValueError("Expected embeddings_dir to exist")

        embedding_datasets = []

        for model1_to_model2_dir in stitched_embeddings_dir.iterdir():
            for dataset_dir in model1_to_model2_dir.iterdir():
                for arch_dir in dataset_dir.iterdir():
                    train_embeddings_file = arch_dir / "embeddings_corpus_train.safetensors"
                    test_embeddings_file = arch_dir / "embeddings_corpus_test.safetensors"

                    if not train_embeddings_file.exists() or test_embeddings_file.exists():
                        assert False

                    model_1, model_2 = model1_to_model2_dir.name.split("_to_")
                    dataset, arch = dataset_dir.name, arch_dir.name

                    train_experiment = ExperimentConfig(
                        dataset_name=dataset,
                        dataset_split="train",
                        source=None,
                    )

                    # stitched_embeddings = EmbeddingDatasetInformation(
                    #     embedding_model_name=model_2,
                    #     embedding_model_type="huggingface"
                    #     if "huggingface" in native_embeddings_dir.name
                    #     else "openai",
                    #     embedding_dimension=model2model_dimension(model_2),
                    #     text_dataset_name=dataset,
                    #     text_dataset_source="huggingface",
                    #     dataset_split=None,
                    #     if "huggingface" in native_embeddings_dir.name else "openai",
                    #     dataset_filepath=embeddings_file,
                    # )

                    # embedding_datasets.append(embedding_dataset)

        return embedding_datasets


# %%
if __name__ == "__main__":
    ### GET TRAINING LOGS ###
    # Save
    train_loss_dir = PROJ_ROOT / "data" / "arguana_loss"
    train_logs_dir = PROJ_ROOT / "data" / "arguana_loss_logs"
    if train_logs_dir.exists():
        train_values = ModelGenerator.load_data_as_train_logs(train_logs_dir / "arguana_loss_logs.json")
        train_values = train_values.model_dump()["logs"]
    else:
        train_logs_dir.mkdir(parents=True, exist_ok=False)
        train_values = ModelGenerator.get_train_logs(PROJ_ROOT / "data" / "arguana_loss")
        anal_dump(TrainingLogs(logs=train_values), "arguana_loss_logs", "data/arguana_loss_logs")

    logger.info(f"Received {len(train_values)} training values")
    matrix = ModelGenerator.get_mapping_from_train_values(train_values)
    labels = sorted(matrix.keys())

    ### MSE MATRIX ###
    mse_matrix = ModelGenerator.get_mse_matrix_from_matrix(matrix, labels)
    text_dataset_name = "arguana"
    fig = visualize_heatmap(
        matrix=mse_matrix,
        config={
            "row_labels": labels,
            "col_labels": labels,
            "title": f"MSE Matrix on {text_dataset_name}",
            "xaxis_title": "Native Embedding Space",
            "yaxis_title": "Target Embedding Space",
        },
    )

    fig["layout"]["yaxis"]["autorange"] = "reversed"
    fig.update_xaxes(side="top")

    save_figure(fig, f"mse_matrix_on{text_dataset_name}", output_dir=PROJ_ROOT / "data" / "figs")
    logger.info(mse_matrix)

    ### MAE MATRIX ###
    mae_matrix = ModelGenerator.get_mae_matrix_from_matrix(matrix, labels)
    text_dataset_name = "arguana"
    fig = visualize_heatmap(
        matrix=mae_matrix,
        config={
            "row_labels": labels,
            "col_labels": labels,
            "title": f"MAE Matrix on {text_dataset_name}",
            "xaxis_title": "Native Embedding Space",
            "yaxis_title": "Target Embedding Space",
        },
    )

    fig["layout"]["yaxis"]["autorange"] = "reversed"
    fig.update_xaxes(side="top")

    save_figure(fig, f"mae_matrix_on{text_dataset_name}", output_dir=PROJ_ROOT / "data" / "figs")
    logger.info(mae_matrix)


# %%
# no org
Path("embeddings") / "dataset" / "model" / "train or validation" / "linear.safetensors"

"""Creates StitchSummaries from a given directory which is the format that Adriano saved the stuff in."""

from pathlib import Path
import json

from loguru import logger

from src.collection_utils import model2model_dimension
from src.schema.training_schemas import EmbeddingDatasetInformation
from src.schema.training_schemas import ExperimentConfig
from src.schema.training_schemas import StitchSummary
from src.utils.general_setup import setup
from src.logic.anal_dump import anal_dump


setup("StitchSummaryGenerator")

PROJ_ROOT = Path(__file__).parent.parent


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
    def fuck_just_get_mse_matrix(dir: Path) -> list[dict]:
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
            info_path = stitch_dir / "stitch_info_pairs.json"
            log_path = stitch_dir / "log.jsonl"

            # Skip if either file is missing
            if not info_path.exists() or not log_path.exists():
                logger.warning(f"Skipping {stitch_dir.name} - missing required files")
                continue

            try:
                # Read the info file
                with open(info_path) as f:
                    info = json.loads(f.read())

                # Read the last line of the log file
                with open(log_path) as f:
                    # Skip to the end and read last line
                    for line in f:
                        last_log = json.loads(line)

                train_value = {
                    "info": info,
                    "log": last_log,
                    "dir": str(stitch_dir),  # Include directory for reference
                }
                train_values.append(train_value)

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in {stitch_dir.name}")
            except Exception as e:
                logger.error(f"Error processing {stitch_dir.name}: {str(e)}")

        return train_values

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


if __name__ == "__main__":
    # embeddings_dir = PROJ_ROOT / "data" / "huggingface_embeddings"
    # dataset_infos = ModelGenerator.native_embedding_dataset_info_from_dir(embeddings_dir)
    # for i, dataset_info in enumerate(dataset_infos):
    #     anal_dump(dataset_info, filename=str(i), output_dir="data/dataset_infos")
    # logger.info(dataset_infos)
    train_values = ModelGenerator.fuck_just_get_mse_matrix(PROJ_ROOT / "data" / "arguana_loss")
    print(train_values)


# no org
Path("embeddings") / "dataset" / "model" / "train or validation" / "linear.safetensors"

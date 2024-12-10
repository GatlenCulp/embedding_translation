"""Generates example StitchSummary and corresponding mock safetensor files."""

from datetime import datetime
from pathlib import Path

from loguru import logger

from src.logic.anal_dump import anal_dump
from src.schema.mock.safe_tensor import generate_mock_safetensor
from src.schema.training_schemas import EmbeddingDatasetInformation
from src.schema.training_schemas import ExperimentConfig
from src.schema.training_schemas import IngestionSettings
from src.schema.training_schemas import StitchEvaluation
from src.schema.training_schemas import StitchEvaluationLog
from src.schema.training_schemas import StitchSummary
from src.schema.training_schemas import TrainSettings
from src.schema.training_schemas import TrainStatus
from src.utils.general_setup import setup


rng = setup("stitch_summary")


def create_example_stitch_summary(
    base_path: str | Path = "data/embeddings", create_safetensors: bool = True
) -> StitchSummary:
    """Create a mock StitchSummary and generate corresponding safetensor files.

    :param base_path: Base directory for storing mock embedding files
    :return: Populated StitchSummary with valid file paths
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # Create mock embedding dataset info with actual files
    source_embeddings = EmbeddingDatasetInformation(
        embedding_model_name="text-embedding-ada-002",
        embedding_model_type="openai",
        embedding_dimension=1536,
        text_dataset_name="hotpotqa",
        chromadb_collection_name="hotpotqa_source_embeddings",
        ingestion_settings=IngestionSettings(),
        dataset_filepath=str(base_path / "source_embeddings.safetensors"),
        collections_filepath=str(base_path / "source_collections"),
    )

    if create_safetensors:
        # Generate source embeddings file
        generate_mock_safetensor(
            shape=(
                1000,
                source_embeddings.embedding_dimension,
            ),  # 1000 embeddings of dimension 1536
            output_path=source_embeddings.dataset_filepath,
            tensor_names=["embeddings"],
        )

    target_embeddings = EmbeddingDatasetInformation(
        embedding_model_name="instructor-xl",
        embedding_model_type="huggingface",
        embedding_dimension=768,
        text_dataset_name="hotpotqa",
        chromadb_collection_name="hotpotqa_target_embeddings",
        ingestion_settings=IngestionSettings(),
        dataset_filepath=str(base_path / "target_embeddings.safetensors"),
        collections_filepath=str(base_path / "target_collections"),
    )

    if create_safetensors:
        # Generate target embeddings file
        generate_mock_safetensor(
            shape=(1000, target_embeddings.embedding_dimension),  # 1000 embeddings of dimension 768
            output_path=target_embeddings.dataset_filepath,
            tensor_names=["embeddings"],
        )

    # Create mock stitched embeddings with actual files
    stitched_train_embeddings = EmbeddingDatasetInformation(
        embedding_model_name="text-embedding-ada-002",
        embedding_model_type="openai",
        embedding_dimension=768,  # Matches target dimension
        text_dataset_name="hotpotqa",
        chromadb_collection_name="hotpotqa_stitched_train",
        ingestion_settings=IngestionSettings(),
        dataset_filepath=str(base_path / "stitched_train_embeddings.safetensors"),
        collections_filepath=str(base_path / "stitched_train_collections"),
        stitch_model_name="stitch_ada002_to_instructor",
    )

    if create_safetensors:
        # Generate stitched train embeddings file
        generate_mock_safetensor(
            shape=(
                1000,
                stitched_train_embeddings.embedding_dimension,
            ),  # 1000 embeddings of dimension 768
            output_path=stitched_train_embeddings.dataset_filepath,
            tensor_names=["embeddings"],
        )

    stitched_test_embeddings = EmbeddingDatasetInformation(
        embedding_model_name="text-embedding-ada-002",
        embedding_model_type="openai",
        embedding_dimension=768,  # Matches target dimension
        text_dataset_name="hotpotqa",
        chromadb_collection_name="hotpotqa_stitched_test",
        ingestion_settings=IngestionSettings(),
        dataset_filepath=str(base_path / "stitched_test_embeddings.safetensors"),
        collections_filepath=str(base_path / "stitched_test_collections"),
        stitch_model_name="stitch_ada002_to_instructor",
    )

    if create_safetensors:
        # Generate stitched test embeddings file
        generate_mock_safetensor(
            shape=(
                200,
                stitched_test_embeddings.embedding_dimension,
            ),  # 200 test embeddings of dimension 768
            output_path=stitched_test_embeddings.dataset_filepath,
            tensor_names=["embeddings"],
        )

    # Create mock experiment configs
    train_experiment = ExperimentConfig(
        dataset_name="hotpotqa",
        dataset_split="train",
        dataset_size=1000,
        source=source_embeddings,
        target=target_embeddings,
        architecture="affine",
        architecture_config={"bias": True},
    )

    test_experiment = ExperimentConfig(
        dataset_name="hotpotqa",
        dataset_split="test",
        dataset_size=200,
        source=source_embeddings,
        target=target_embeddings,
        architecture="affine",
        architecture_config={"bias": True},
    )

    # Create mock training results
    train_status = TrainStatus(
        num_epochs=10,
        num_embeddings_per_epoch=1000,
        num_embeddings_trained_on_total=10000,
        status="completed",
        wandb_run_project="stitch_training",
        wandb_run_name="ada002_to_instructor_001",
        time_start=datetime(2024, 3, 15, 10, 0),
        time_end=datetime(2024, 3, 15, 11, 0),
        time_taken_sec=3600,
        time_taken_min=60,
        time_taken_hr="1.0",
    )

    # Create mock evaluation logs
    train_eval_log = StitchEvaluationLog(
        evaluations=[
            StitchEvaluation(
                epoch_num=i,
                num_embeddings_evaluated=1000,
                stitching_mse=0.5 - (0.04 * i),  # Decreasing MSE
                stitching_mae=0.4 - (0.03 * i),  # Decreasing MAE
                evaluation_data_split="train",
            )
            for i in range(10)
        ]
    )

    test_eval_log = StitchEvaluationLog(
        evaluations=[
            StitchEvaluation(
                epoch_num=9,  # Final epoch only
                num_embeddings_evaluated=200,
                stitching_mse=0.15,
                stitching_mae=0.12,
                evaluation_data_split="test",
            )
        ]
    )

    # Create the complete StitchSummary
    mock_stitch_summary = StitchSummary(
        training_experiment_config=train_experiment,
        train_settings=TrainSettings(experiment_settings=train_experiment),
        training_evaluation_log=train_eval_log,
        train_status_final=train_status,
        train_stitch_embeddings=stitched_train_embeddings,
        test_experiment_config=test_experiment,
        test_evaluation_log=test_eval_log,
        test_stitch_embeddings=stitched_test_embeddings,
    )

    return mock_stitch_summary


def create_inverse_stitch_summary(stitch_summary: StitchSummary) -> StitchSummary:
    """Creates a mock inverse stitch by reversing the source and target embeddings.

    :param stitch_summary: Original StitchSummary to invert
    :return: New StitchSummary with source/target reversed and appropriate adjustments
    """
    # Create deep copy of original summary
    inv_stitch = stitch_summary.model_copy(deep=True)

    # Swap source and target in training experiment
    (
        inv_stitch.training_experiment_config.source,
        inv_stitch.training_experiment_config.target,
    ) = (
        inv_stitch.training_experiment_config.target,
        inv_stitch.training_experiment_config.source,
    )

    # Swap source and target in test experiment
    (
        inv_stitch.test_experiment_config.source,
        inv_stitch.test_experiment_config.target,
    ) = (
        inv_stitch.test_experiment_config.target,
        inv_stitch.test_experiment_config.source,
    )

    # Assume stitching happened perfectly and substitute target for stitch_embeddings
    inv_stitch.train_stitch_embeddings = inv_stitch.training_experiment_config.target
    inv_stitch.test_stitch_embeddings = inv_stitch.test_experiment_config.target

    return inv_stitch


def save_inverse_stitch(stitch_summary: StitchSummary, suffix: str = "") -> StitchSummary:
    """Creates a mock inverse stitch by reversing the source and target embeddings.

    :param stitch_summary: Original StitchSummary to invert
    :return: New StitchSummary with source/target reversed and appropriate adjustments
    """
    inv_stitch = create_inverse_stitch_summary(stitch_summary)
    anal_dump(
        inv_stitch,
        output_dir="data/stitch_summaries",
        filename=f"{inv_stitch.slug}{suffix}",
    )

    return inv_stitch


def save_mock_stitch(suffix: str = "") -> StitchSummary:
    """Generates a mock StitchSummary to data using the name `{slug}{suffix}.json`."""
    mock_stitch_summary = create_example_stitch_summary(create_safetensors=True)
    validated_stitch_summary = StitchSummary.model_validate(mock_stitch_summary)
    logger.info(validated_stitch_summary)
    anal_dump(
        validated_stitch_summary,
        output_dir="data/stitch_summaries",
        filename=f"{validated_stitch_summary.slug}{suffix}",
    )
    return validated_stitch_summary


def main() -> None:
    for suffix in ("_01", "_02", "_03"):
        stitch_summary = save_mock_stitch(suffix=suffix)
        _ = save_inverse_stitch(stitch_summary, suffix=suffix)


if __name__ == "__main__":
    main()

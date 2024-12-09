"""Generates example StitchSummary."""

from datetime import datetime

from src.schema.training_schemas import EmbeddingDatasetInformation
from src.schema.training_schemas import ExperimentConfig
from src.schema.training_schemas import IngestionSettings
from src.schema.training_schemas import StitchEvaluation
from src.schema.training_schemas import StitchEvaluationLog
from src.schema.training_schemas import StitchSummary
from src.schema.training_schemas import TrainSettings
from src.schema.training_schemas import TrainStatus


def create_example_stitch_summary() -> StitchSummary:
    # Create mock embedding dataset info
    source_embeddings = EmbeddingDatasetInformation(
        embedding_model_name="text-embedding-ada-002",
        embedding_model_type="openai",
        embedding_dimension=1536,
        text_dataset_name="hotpotqa",
        chromadb_collection_name="hotpotqa_source_embeddings",
        ingestion_settings=IngestionSettings(),
        dataset_filepath="/path/to/source/embeddings.jsonl",
        collections_filepath="/path/to/source/collections",
    )

    target_embeddings = EmbeddingDatasetInformation(
        embedding_model_name="instructor-xl",
        embedding_model_type="huggingface",
        embedding_dimension=768,
        text_dataset_name="hotpotqa",
        chromadb_collection_name="hotpotqa_target_embeddings",
        ingestion_settings=IngestionSettings(),
        dataset_filepath="/path/to/target/embeddings.jsonl",
        collections_filepath="/path/to/target/collections",
    )

    # Create mock stitched embeddings
    stitched_train_embeddings = EmbeddingDatasetInformation(
        embedding_model_name="text-embedding-ada-002",
        embedding_model_type="openai",
        embedding_dimension=768,  # Matches target dimension
        text_dataset_name="hotpotqa",
        chromadb_collection_name="hotpotqa_stitched_train",
        ingestion_settings=IngestionSettings(),
        dataset_filepath="/path/to/stitched/train_embeddings.jsonl",
        collections_filepath="/path/to/stitched/train_collections",
        stitch_model_name="stitch_ada002_to_instructor",
    )

    stitched_test_embeddings = EmbeddingDatasetInformation(
        embedding_model_name="text-embedding-ada-002",
        embedding_model_type="openai",
        embedding_dimension=768,  # Matches target dimension
        text_dataset_name="hotpotqa",
        chromadb_collection_name="hotpotqa_stitched_test",
        ingestion_settings=IngestionSettings(),
        dataset_filepath="/path/to/stitched/test_embeddings.jsonl",
        collections_filepath="/path/to/stitched/test_collections",
        stitch_model_name="stitch_ada002_to_instructor",
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


def main() -> None:
    mock_stitch_summary = create_example_stitch_summary()
    StitchSummary.model_validate(mock_stitch_summary)
    print(mock_stitch_summary)


if __name__ == "__main__":
    main()

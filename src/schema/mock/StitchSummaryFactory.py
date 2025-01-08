"""Creates mock data using StitchSummary.

TODO: Get this to work
"""

from datetime import datetime
from datetime import timedelta
from typing import Any

from polyfactory.factories import BaseFactory

from src.schema.training_schemas import EmbeddingDatasetInformation
from src.schema.training_schemas import ExperimentConfig
from src.schema.training_schemas import StitchEvaluation
from src.schema.training_schemas import StitchEvaluationLog
from src.schema.training_schemas import StitchSummary
from src.schema.training_schemas import TrainSettings
from src.schema.training_schemas import TrainStatus


class EmbeddingDatasetInformationFactory(BaseFactory[EmbeddingDatasetInformation]):
    """Factory for generating test EmbeddingDatasetInformation instances."""

    __model__ = EmbeddingDatasetInformation

    @classmethod
    def get_default_embedding_model_name(cls) -> str:
        return cls.random_choice(["bert-base-uncased", "gpt2", "roberta-base"])

    @classmethod
    def get_default_text_dataset_name(cls) -> str:
        return cls.random_choice(["hotpotqa", "squad", "nq"])

    @classmethod
    def get_default_chromadb_collection_name(cls) -> str:
        return f"collection_{cls.random_int(1000, 9999)}"


class ExperimentConfigFactory(BaseFactory[ExperimentConfig]):
    """Factory for generating test ExperimentConfig instances."""

    __model__ = ExperimentConfig

    @classmethod
    def get_default_dataset_name(cls) -> str:
        return cls.random_choice(["HotPotQA", "SQuAD", "NaturalQuestions"])

    @classmethod
    def get_default_dataset_size(cls) -> int:
        return cls.random_int(1000, 10000)


class StitchEvaluationFactory(BaseFactory[StitchEvaluation]):
    """Factory for generating test StitchEvaluation instances."""

    __model__ = StitchEvaluation

    @classmethod
    def get_default_stitching_mse(cls) -> float:
        return cls.random_float(0.001, 0.1)

    @classmethod
    def get_default_stitching_mae(cls) -> float:
        return cls.random_float(0.01, 0.2)


class StitchSummaryFactory(BaseFactory[StitchSummary]):
    """Factory for generating test StitchSummary instances."""

    __model__ = StitchSummary

    @classmethod
    def create_training_evaluation_log(cls) -> StitchEvaluationLog:
        """Creates a realistic training evaluation log."""
        num_epochs = 10
        evaluations = []
        for epoch in range(num_epochs):
            eval_instance = StitchEvaluationFactory.build(
                epoch_num=epoch,
                num_embeddings_evaluated=1000 * (epoch + 1),
                stitching_mse=0.1 / (epoch + 1),  # Decreasing MSE
                stitching_mae=0.2 / (epoch + 1),  # Decreasing MAE
            )
            evaluations.append(eval_instance)
        return StitchEvaluationLog(evaluations=evaluations)

    @classmethod
    def create_train_status(cls) -> TrainStatus:
        """Creates a realistic train status."""
        start_time = datetime.now() - timedelta(hours=2)
        return TrainStatus(
            num_epochs=10,
            num_embeddings_per_epoch=1000,
            num_embeddings_trained_on_total=10000,
            status="completed",
            time_start=start_time,
            time_end=datetime.now(),
            time_taken_sec=7200,
            time_taken_min=120,
            time_taken_hr="2.0",
            wandb_run_project="stitch_training",
            wandb_run_name=f"run_{cls.random_int(1000, 9999)}",
        )

    @classmethod
    def build(cls, **kwargs: Any) -> StitchSummary:
        """Build a StitchSummary instance with realistic related data."""
        # Create source and target embedding datasets
        source_embeddings = EmbeddingDatasetInformationFactory.build()
        target_embeddings = EmbeddingDatasetInformationFactory.build()

        # Create experiment configs
        train_config = ExperimentConfigFactory.build(
            source=source_embeddings,
            target=target_embeddings,
        )
        test_config = ExperimentConfigFactory.build(
            source=source_embeddings,
            target=target_embeddings,
            dataset_split="test",
        )

        # Build the complete StitchSummary
        return super().build(
            training_experiment_config=train_config,
            train_settings=TrainSettings(experiment_settings=train_config),
            training_evaluation_log=cls.create_training_evaluation_log(),
            train_status_final=cls.create_train_status(),
            train_stitch_embeddings=EmbeddingDatasetInformationFactory.build(
                stitch_model_name="stitch_v1"
            ),
            test_experiment_config=test_config,
            test_evaluation_log=StitchEvaluationLog(
                evaluations=[StitchEvaluationFactory.build()]
            ),
            test_stitch_embeddings=EmbeddingDatasetInformationFactory.build(
                stitch_model_name="stitch_v1"
            ),
            **kwargs,
        )


def main() -> None:
    return StitchSummaryFactory.build()


if __name__ == "__main__":
    main()

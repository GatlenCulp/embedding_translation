import os

from dotenv import load_dotenv
from flask import Flask

from owler_fork.owlergpt import utils
from owler_fork.owlergpt.modern.schemas import EmbeddingDatasetInformation
from owler_fork.owlergpt.modern.schemas import EmbeddingMetadata
from owler_fork.owlergpt.modern.schemas import EvaluationSettings
from owler_fork.owlergpt.modern.schemas import ExperimentConfig
from owler_fork.owlergpt.modern.schemas import IngestionSettings
from owler_fork.owlergpt.modern.schemas import StitchEvaluation
from owler_fork.owlergpt.modern.schemas import StitchEvaluationLog
from owler_fork.owlergpt.modern.schemas import StitchSummary
from owler_fork.owlergpt.modern.schemas import TrainSettings
from owler_fork.owlergpt.modern.schemas import TrainStatus


load_dotenv()


def create_app():
    """App factory (https://flask.palletsprojects.com/en/2.3.x/patterns/appfactories/)"""
    app = Flask(__name__)

    # Note: Every module in this app assumes the app context is available and initialized.
    with app.app_context():
        utils.check_env()

        os.makedirs(os.environ["DATASET_FOLDER_PATH"], exist_ok=True)
        os.makedirs(os.environ["VISUALIZATIONS_FOLDER_PATH"], exist_ok=True)
        os.makedirs(os.environ["VECTOR_SEARCH_PATH"], exist_ok=True)
        os.makedirs(os.environ["EVAL_FOLDER_PATH"], exist_ok=True)

        from owlergpt import commands

        return app

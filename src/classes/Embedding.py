"""Stores the schema for how to organize data for the pipeline."""

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from src.utils.general_setup import setup


setup("classes")


class Embedding(BaseModel):
    """An embedding from an arbitrary model."""

    vector: list[float] = Field()

    model_config = ConfigDict(extra="forbid")

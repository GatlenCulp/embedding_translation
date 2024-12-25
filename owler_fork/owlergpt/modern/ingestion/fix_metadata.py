from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

from chromadb.api.models.Collection import Collection
from pydantic import BaseModel
from pydantic import ValidationError

from owlergpt.modern.collection_utils import OPENAI_MODELS
from owlergpt.modern.collection_utils import model2model_dimension
from owlergpt.modern.collection_utils import parse_collection_name
from owlergpt.modern.schemas import EmbeddingDatasetInformation
from owlergpt.modern.schemas import EmbeddingMetadata
from owlergpt.modern.schemas import IngestionSettings


""" XXX
What needs to be done:
1. Split the dataset into train and test in a consistent way across all collections
2. Add all the pre-requisite metadata to all the embeddings (i.e. each chunk should have a document id as it does, a 
    chunk id (as it doesn't), split, and embedding, plus chunk text prolly) - `EmbeddingMetadata`
3. Make sure the metadata for the dataset overall is correct
4. Validate that ALL the different models' collections are present

=> tests
=> actually modify the datasets
(cool after this everything we have should be properly tagged with the metadata)

Then finish implementing the trainer and make sure that it spits out the right types of metadata tags and saves to the filesystem OK
=> launch and sleep
=> once the shit is done we need to write the data filtration (small extension of the shit from above)
    => create the new datasets
=> enumerate datasets and relationships with desired comparisons
=> understand the visualization pipeline and then automatically generate the desired visualizations
=> need to be able to launch more training jobs

ALWAYS REMEMBER THAT THIS SHIT HAS TO BE TESTABLE

"""


class EmbeddingMetadataPopulatorArgs(BaseModel):
    target_split_frac: float
    overwrite_metadata: bool = False


class EmbeddingMetadataPopulatorIdempotencyBug(Exception):
    """Exception to be raised if the metadata is already properly setup."""


class EmbeddingMetadataPopulatorCoordinator:
    """Helper to coordinate the train-test split across all collections for a given dataset. It is just a container
    that is used to store a map of train-test split for each collection (where the ID should be defined based on chunk
    and used to map to everything except the embedding). This object also remembers if it was already populated or not
    and if it was it tells you to use the previous values and otherwise it creates new ones.

    The proper usage is to:
    1. Validate collections (datasets)
        - Validates sizes
        - Validates that either all or none have been populated (alternative not supported yet)
    2. Populate the train-test split for each collection. Provide an idempotent interface

    TODO(Adriano) it would be nice to have the chunks be ordered?
    """

    def __init__(
        self,
        selected_folder: str,
        collections: list[Collection],
        metadata_populator_args: EmbeddingMetadataPopulatorArgs,
    ):
        self.selected_folder: str = selected_folder
        self.collections: list[Collection] = collections
        self.metadata_populator_args: EmbeddingMetadataPopulatorArgs = (
            metadata_populator_args
        )
        # Return the embedding metadata to be stored for this element in all the collections, says
        # - chunk_text (redundant)
        # - chunk_id (new)
        # - record_id (already there/redundant)
        # - record_text (redundant)
        # - record_type (new)
        # - record_split (new)
        # - tags (new)
        self.already_populated_one_collection: bool = False
        self.chunk_text2embedding_metadata: dict[str, EmbeddingMetadata] = {}
        self.idx2embedding_metadata: dict[int, EmbeddingMetadata] = {}

    def __collection_metadata_is_populated(self, collection: Collection) -> bool:
        """Internal helper."""
        try:
            EmbeddingMetadata.model_validate(collection.metadata)
            return True
        except ValidationError:
            return False

    def __entry_metadata_is_populated(self, entry: dict[str, Any]) -> bool:
        """Internal helper."""
        try:
            EmbeddingMetadata.model_validate(entry)
            return True
        except ValidationError:
            return False

    def get_embedding_metadata(self, chunk_metadata: str) -> list[EmbeddingMetadata]:
        """Coordinating helper: given a chunk text, return the embedding metadata for it."""
        chunk_text = chunk_metadata[
            "record_text"
        ]  # NOTE by this they SEEM to actually mean chunk text???
        record_id = chunk_metadata["record_id"]
        if self.already_populated_one_collection:
            assert (
                chunk_text in self.chunk_text2embedding_metadata
            ), f"Chunk text not found in metadata: {chunk_text}"
            return self.chunk_text2embedding_metadata[chunk_text]
        assert (
            chunk_text not in self.chunk_text2embedding_metadata
        ), f"Chunk text should not be in metadata: {chunk_text}"
        tags = {}
        return [
            EmbeddingMetadata(
                chunk_id=str(uuid4()),
                chunk_text=chunk_text,
                record_id=record_id,
                record_text=record_text,
                record_type=record_type,
                record_split=record_split,
                tags=tags,
            )
        ]

    def validate_collections(self):
        """Validate the datasets."""
        # 1. validate sizes
        sizes = [len(collection.get()) for collection in self.collections]
        assert all(
            size == sizes[0] for size in sizes
        ), f"All collections must have the same number of elements: {sizes}"
        # 2. validate that either all or none have been populated (collection granularity)
        metadata_populated = [
            self.__collection_metadata_is_populated(collection)
            for collection in self.collections
        ]
        assert (
            all(metadata_populated) or not any(metadata_populated)
        ), f"Either all or none collections must have been populated: {metadata_populated}"
        # 3. validate that the train-test split is consistent across collections
        all_is_popped: list[bool] = []
        for collection in self.collections:
            is_popped: bool = any(
                self.__entry_metadata_is_populated(metadatas)
                for _, metadatas in collection.get()["metadatas"]
            )
            all_is_popped.append(is_popped)
        all_is_popped_dump = "\n".join(
            [
                f"{collection.name}: {is_popped}"
                for collection, is_popped in zip(
                    self.collections, all_is_popped, strict=False
                )
            ]
        )
        assert (
            all(all_is_popped) or not any(all_is_popped)
        ), f"Either all or none collections must have been populated: {all_is_popped_dump}"

    def preprocess_idx2embedding_metadata(self):
        """Preprocess the idx2embedding_metadata to be able to use it to populate the metadata."""
        raise NotImplementedError("Not implemented")  # XXX

    def populate_train_test_splits(self):
        """Populate the train-test split for each collection."""
        initializer_collection = self.collections[0]
        other_collections = self.collections[1:]

        # 1. Populate the initializer collection
        populator = EmbeddingMetadataPopulator(
            initializer_collection,
            self.selected_folder,
            self.metadata_populator_args,
            self,
        )
        populator.populate_metadata()
        self.already_populated_one_collection = True

        # 2. Populate the other collections with switch toggled
        for collection in other_collections:
            populator = EmbeddingMetadataPopulator(
                collection,
                self.selected_folder,
                self.metadata_populator_args,
                self,
            )
            populator.populate_metadata()

    def main(self):
        """Main method to validate and populate the train-test split."""
        self.validate_collections()
        self.preprocess_idx2embedding_metadata()
        self.populate_train_test_splits()


class EmbeddingMetadataPopulator:
    """Idempotent class to "populate" the metadata for a given collection. This involves
    - Populating the collection metadata to trace the lineage of this collection
        (only supports text -> embedding datsets, not embeddings -> embeddings)
    - Mark data-entries as train or test effectively with coordination help from
        a dataset `EmbeddingMetadataPopulatorCoordinator` (which makes sure that the same
        elements are marked as train or test across all collections for this
        specific dataset).
    - Add in unique IDs per data-entry so that it's easy to later be able to
        trace back to the original data-entry.
    - If metadata is already properly setup then it will NOT do this shit
    """

    def __init__(
        self,
        collection: Collection,
        selected_folder: str,
        metadata_populator_args: EmbeddingMetadataPopulatorArgs,
        coordinator: EmbeddingMetadataPopulatorCoordinator,
    ):
        self.collection = collection
        self.selected_folder = selected_folder
        self.metadata_populator_args = metadata_populator_args
        self.coordinator = coordinator

    def infer_ingestion_settings(self) -> IngestionSettings:
        """Helper to infer the ingestion settings that you used for the semantic search.

        Usually you have a fixed os.environ set that you get from your .env file
        and this does not vary, so we try to use that to infer what parameters
        you used.
        """
        # 1. assert supported modes are the used ones
        if "chunk_preprocessing_mode" in self.collection.metadata:
            assert self.collection.metadata["chunk_preprocessing_mode"] in ["add_prefix"], f"Unsupported chunk preprocessing mode: {self.collection.metadata['chunk_preprocessing_mode']}"  # fmt: skip
        if "query_preprocessing_mode" in self.collection.metadata:
            assert self.collection.metadata["query_preprocessing_mode"] in ["add_prefix"], f"Unsupported query preprocessing mode: {self.collection.metadata['query_preprocessing_mode']}"  # fmt: skip

        # 2. Fetch all ensuring consistency (environ CAN be none)
        chunk_size = int(os.environ["VECTOR_SEARCH_SENTENCE_DEFAULT_CHUNK_SIZE"])
        if "chunk_size" in self.collection.metadata:
            assert chunk_size is None or self.collection.metadata["chunk_size"] == chunk_size, f"Chunk size mismatch: {self.collection.metadata['chunk_size']} != {chunk_size} (disable environ?)"  # fmt: skip
            chunk_size = self.collection.metadata["chunk_size"]
        device = None  # <--- note important
        distance_function = os.environ["VECTOR_SEARCH_DISTANCE_FUNCTION"]
        if "distance_function" in self.collection.metadata:
            assert distance_function is None or self.collection.metadata["distance_function"] == distance_function, f"Distance function mismatch: {self.collection.metadata['distance_function']} != {distance_function} (disable environ?)"  # fmt: skip
            distance_function = self.collection.metadata["distance_function"]
        normalize_embeddings = os.environ["VECTOR_SEARCH_NORMALIZE_EMBEDDINGS"]
        if "normalize_embeddings" in self.collection.metadata:
            assert normalize_embeddings is None or self.collection.metadata["normalize_embeddings"] == normalize_embeddings, f"Normalize embeddings mismatch: {self.collection.metadata['normalize_embeddings']} != {normalize_embeddings} (disable environ?)"  # fmt: skip
            normalize_embeddings = self.collection.metadata["normalize_embeddings"]
        chunk_prefix = os.environ["VECTOR_SEARCH_CHUNK_PREFIX"]
        if "chunk_prefix" in self.collection.metadata:
            assert chunk_prefix is None or self.collection.metadata["chunk_prefix"] == chunk_prefix, f"Chunk prefix mismatch: {self.collection.metadata['chunk_prefix']} != {chunk_prefix} (disable environ?)"  # fmt: skip
            chunk_prefix = self.collection.metadata["chunk_prefix"]
        query_prefix = os.environ["VECTOR_SEARCH_QUERY_PREFIX"]
        if "query_prefix" in self.collection.metadata:
            assert query_prefix is None or self.collection.metadata["query_prefix"] == query_prefix, f"Query prefix mismatch: {self.collection.metadata['query_prefix']} != {query_prefix} (disable environ?)"  # fmt: skip
            query_prefix = self.collection.metadata["query_prefix"]
        chunk_overlap = int(os.environ["VECTOR_SEARCH_CHUNK_OVERLAP"])
        if "chunk_overlap" in self.collection.metadata:
            assert chunk_overlap is None or self.collection.metadata["chunk_overlap"] == chunk_overlap, f"Chunk overlap mismatch: {self.collection.metadata['chunk_overlap']} != {chunk_overlap} (disable environ?)"  # fmt: skip
            chunk_overlap = self.collection.metadata["chunk_overlap"]
        dataloader_batch_size = int(os.environ["BATCH_SIZE"])
        if "dataloader_batch_size" in self.collection.metadata:
            assert dataloader_batch_size is None or self.collection.metadata["dataloader_batch_size"] == dataloader_batch_size, f"Dataloader batch size mismatch: {self.collection.metadata['dataloader_batch_size']} != {dataloader_batch_size} (disable environ?)"  # fmt: skip
            dataloader_batch_size = self.collection.metadata["dataloader_batch_size"]
        dataloader_num_workers = 4  # default hardcoded
        if "hnsw:space" in self.collection.metadata:
            assert (
                distance_function == self.collection.metadata["hnsw:space"]
            ), f"Distance function mismatch: {distance_function} != {self.collection.metadata['hnsw:space']} (disable environ?)"
            distance_function = self.collection.metadata["hnsw:space"]
        assert (
            device not in self.collection.metadata
        ), f"Device should not be in metadata: {self.collection.metadata} (reserved name, but not used rn)"
        if "normalize_embeddings" in self.collection.metadata:
            normalize_embeddings = self.collection.metadata["normalize_embeddings"]

        # 3. MAke sure this shit is non-None always
        assert chunk_size is not None
        assert distance_function is not None
        assert normalize_embeddings is not None
        assert chunk_prefix is not None
        assert query_prefix is not None
        assert chunk_overlap is not None
        assert dataloader_batch_size is not None
        assert dataloader_num_workers is not None

        return IngestionSettings(
            chunk_size=chunk_size,
            device=device,
            distance_function=distance_function,
            normalize_embeddings=normalize_embeddings,
            chunk_preprocessing_mode="add_prefix",
            query_preprocessing_mode="add_prefix",
            chunk_prefix=chunk_prefix,
            query_prefix=query_prefix,
            chunk_overlap=chunk_overlap,
            dataloader_batch_size=dataloader_batch_size,
            dataloader_num_workers=dataloader_num_workers,
        )

    def infer_embedding_dataset_information(self) -> EmbeddingDatasetInformation:
        """Helper to infer the embedding dataset information that you used for the semantic search."""
        current_metadata = self.collection.metadata
        try:
            obj = EmbeddingDatasetInformation.model_validate(current_metadata)
            if not self.metadata_populator_args.overwrite_metadata:
                raise EmbeddingMetadataPopulatorIdempotencyBug(
                    "Embedding collection metadata already properly setup"
                )
            return obj
        except ValidationError:
            inferred_ingestion = self.infer_ingestion_settings()
            embedding_model_name, text_dataset, chromadb_collection_name = (
                parse_collection_name(self.collection.name)
            )
            is_openai = embedding_model_name in OPENAI_MODELS
            assert "/" not in embedding_model_name, f"Embedding model name should not contain '/': {embedding_model_name}"  # fmt: skip
            assert text_dataset == self.selected_folder, f"Text dataset mismatch: {text_dataset} != {self.selected_folder}"  # fmt: skip
            return EmbeddingDatasetInformation(
                embedding_model_name=embedding_model_name,
                embedding_model_type=("openai" if is_openai else "huggingface"),
                embedding_dimension=model2model_dimension(embedding_model_name),
                text_dataset_name=text_dataset,
                text_dataset_source="huggingface",
                chromadb_collection_name=chromadb_collection_name,
                ingestion_settings=inferred_ingestion,
            )

    def populate_metadata(self):
        """Primary method of this class."""
        # 1. Update the collection metadata
        try:
            coll_metadata: EmbeddingMetadata = (
                self.infer_embedding_dataset_information()
            )
            self.collection.modify(metadata=coll_metadata.model_dump())
        except EmbeddingMetadataPopulatorIdempotencyBug:
            return  # already properly setup
        # 2. Update the per-chunk metadata by:
        #   2.1 Setting their train-test split setting
        #   2.2 Setting their chunk ids
        for collection in self.collections:
            # def update_collection_metadata(collection: Collection, metadata_key: str, new_value: any):
            #     results = collection.get()
            #     for idx, metadata in enumerate(results['metadatas']):
            #         metadata[metadata_key] = new_value
            #         collection.update(
            #             ids=results['ids'][idx],
            #             metadatas=metadata
            #         ) # XXX
            collection.add(metadatas=metadata)

    @staticmethod
    def populate_metadata_tags(*args, **kwargs):
        """A non-safeguarded method that you could repurpose later to add information
        such as about the semantic classification of documents or chunks. The purpose
        of this method is to be implemented and used later (this class may be moved to
        a location of higher dependency).
        """
        raise NotImplementedError("Not implemented")

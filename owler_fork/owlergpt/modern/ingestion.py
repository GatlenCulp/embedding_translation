from __future__ import annotations
from typing import List, Literal
import os
import click
from owlergpt.utils import JSONDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter, TokenTextSplitter
from chromadb import PersistentClient, Settings

class OriginalIngestion:
    """
    A static class to provide support for (a lot of) the original ingestion logic. This allows
    us to, for example, test. It also has methods to facilitate testing on fake datasets by
    turning hard-coded string datasets into JSONDataset datasets.
    """
    pass # XXX

    def create_collection(
            vector_dataset_path: str,
            selected_folders: List[str],
            tokens_per_chunk: int,
            chunk_overlap: int,
            model_name: str,
            text_splitter: TokenTextSplitter,
            record_type: Literal["query", "document"]
        ) -> None: # XXX
        chroma_client = PersistentClient(
            path=vector_dataset_path,
            settings=Settings(anonymized_telemetry=False),
        )
        admin_client = AdminClient.from_system(chroma_client._system)

        # 1. Create databases
        print(f"Creating {len(selected_folders)} databases")
        db_names = [
            f"{selected_folder}_{tokens_per_chunk}"
            for selected_folder in selected_folders
        ]
        assert len(db_names) == len(selected_folders)
        collections = []
        for db_name, selected_folder in tqdm(zip(db_names, selected_folders), desc="| Creating databases + collections |"):
            try:
                admin_client.create_database(db_name)
                click.echo(f"Created dataset-specific DB {db_name} to store embeddings.")
            except UniqueConstraintError:
                click.echo(f"Dataset-specific DB {db_name} already exists. Using it to store embeddings")
            chroma_client.set_tenant(tenant=DEFAULT_TENANT, database=db_name)
            # Include VECTOR_SEARCH_SENTENCE_TRANSFORMER_MODEL in the collection name
            collection_name = f"{selected_folder}_{transformer_model}_CharacterSplitting_{tokens_per_chunk}"

            try:
                # Attempt to create a new collection with the selected folder name
                chroma_collection = chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": os.environ["VECTOR_SEARCH_DISTANCE_FUNCTION"]},
                )
                collections.append(chroma_collection)
            except UniqueConstraintError:
                # If the collection already exists, delete it and create a new one
                click.echo(f"Collection {collection_name} already exists. Removing and creating a new one.")
                chroma_client.delete_collection(name=collection_name)
                chroma_collection = chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": os.environ["VECTOR_SEARCH_DISTANCE_FUNCTION"]},
                )
                collections.append(chroma_collection)
        assert len(collections) == len(db_names)

        print(f"Processing dataset {selected_folders[0]}")
        total_records = 0
        total_embeddings = 0
        batch_size = int(environ.get("BATCH_SIZE"))

        # Process the batch of documents
        # NOTE: single 
        for filename in ['corpus.jsonl', 'queries.jsonl']:
            if filename == "queries.jsonl":
                record_type = "query" # <---- imporant NOTE
            else:
                record_type = "document"
            dataset = JSONDataset(os.path.join(environ["DATASET_FOLDER_PATH"], selected_folders[0], filename), text_splitter,
                                model_name, tokens_per_chunk, chunk_overlap, environ.get("VECTOR_SEARCH_CHUNK_PREFIX"),
                                record_type)
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
            total_records += dataset.__len__()
            for documents, ids, text_chunks in tqdm(dataloader, desc='| Computing embeddings |', total=len(dataloader)):
                if len(documents) == 0 or len(ids) == 0 or len(text_chunks) == 0:
                    continue
                # Generate embeddings for each chunk
                if model_name in OPENAI_MODELS:
                    embeddings = []
                    data = client.embeddings.create(input=text_chunks, model=model_name).data
                    for entry in data:
                        embeddings.append(entry.embedding)
                elif model_name in COHERE_MODELS:
                    embeddings = client.embed(texts=text_chunks, model=model_name, input_type="search_" + record_type,
                                            embedding_types=['float']).embeddings.float
                else:
                    embeddings = embedding_model.encode(text_chunks, normalize_embeddings=normalize_embeddings).tolist()

                # Prepare metadata for each chunk
                metadatas = [
                    {"record_id": ids[i], "record_text": text_chunks[i], "record_type": record_type}
                    for i in range(len(text_chunks))
                ]

                total_embeddings += len(embeddings)

                # Store embeddings and metadata in the vector store
                chroma_collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

            click.echo(f"Processed {total_records} documents, generated {total_embeddings} embeddings.")
import os
import click
import cohere
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter, TokenTextSplitter
from chromadb import PersistentClient, Settings
from chromadb.api.client import AdminClient
from chromadb.config import DEFAULT_TENANT
from chromadb.db.base import UniqueConstraintError
from flask import current_app
from openai import OpenAI
from owlergpt.utils import JSONDataset, collate_fn, choose_dataset_folders
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import yaml


OPENAI_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]
COHERE_MODELS = ["embed-english-v3.0"]

class FastEmbedder:
    """
    A class to quickly embed a ton of different datasets' and models' worth of text. It's able to
    parallelize across different models and thereby get big speedups. A lot of embedding models are
    only up to ~1GB in size, so we can actually load up to 80 of them and get a crazy level of speedup
    (i.e. if we have 5 datasets and 10 models, we can actually do all 50 combinations in parallel).

    The way this is done is:
    """
    def __init__(self, model_name: str, device: str, parallelism_batch_size: int):
        self.model_name = model_name
        self.device = device
        self.parallelism_batch_size = parallelism_batch_size
    
    def serial_generation(self):
        pass # XXX

    def parallel_generation(self):
        pass # XXX


# XXX this will have to be modified
def create_split_embedding_models(model_name: str, parallelism_batch_size: int) -> List[SentenceTransformer]:
    return [SentenceTransformer(
        model_name, device=os.environ["VECTOR_SEARCH_SENTENCE_TRANSFORMER_DEVICE"],
    ) for _ in range(parallelism_batch_size)] # XXX

@current_app.cli.command("ingest_ds")
def ingest_dataset() -> None:
    environ = os.environ
    models_yaml_path = Path(__file__).parent.parent.parent / "models.yaml"
    assert models_yaml_path.exists(), f"Models file not found at {models_yaml_path}"
    with open(models_yaml_path, "r") as f:
        models = yaml.safe_load(f)
    print("========================= START MODELS", models)
    for model_obj in models["models"]:
        print("========================= MODEL", model_obj["name"])
    print("========================= END MODELS")
    selected_folders = choose_dataset_folders(environ["DATASET_FOLDER_PATH"])
    if len(selected_folders) != 1:
        raise ValueError("Only one dataset folder can be processed at a time")
    for model_obj in models["models"]:
        model_name = model_obj["name"]
        print("========================= INJEESTING FOR MODEL", model_name)
        tokens_per_chunk = 256
        chunk_overlap = int(environ["VECTOR_SEARCH_TEXT_SPLITTER_CHUNK_OVERLAP"])
        normalize_embeddings = environ["VECTOR_SEARCH_NORMALIZE_EMBEDDINGS"] == "true"
    
        parallelism_batch_size = None
        embedding_model: SentenceTransformer | None = None
        if environ["VECTOR_SEARCH_SENTENCE_TRANSFORMER_DEVICE"] == "cpu":
            raise ValueError("CPU is not supported for embedding model. Please use a GPU.")
            # click.echo("Using CPU for embedding model. This might be slow...")
        else:
            print("OkOKOK ===============================================") # XXX
        if model_name not in OPENAI_MODELS and model_name not in COHERE_MODELS:
            parallelism_batch_size = 1
            # parallelism_batch_size = click.prompt("Please enter how many models to generate dbs in parallel for:", type=int, default=8)
            embedding_model = create_split_embedding_models(model_name, parallelism_batch_size)[0]

        # # Ask the user for the tokens_per_chunk value
        # tokens_per_chunk = click.prompt("Please enter the tokens per chunk value (128, 256, 512, 1024)", type=int, default=256)

        # # Check if the entered value is valid
        # if tokens_per_chunk not in [128, 256, 512, 1024]:
        #     click.echo("Invalid tokens per chunk value. Exiting.")
        #     return

        # Use the tokens_per_chunk value when initializing the text_splitter
        click.echo("| Initializing text splitters |")
        if model_name in OPENAI_MODELS:
            text_splitter = TokenTextSplitter(
                model_name=model_name,
                chunk_overlap=chunk_overlap,
                chunk_size=tokens_per_chunk
            )
            transformer_model = model_name
            client = OpenAI(api_key=environ["OPENAI_KEY"])
        elif model_name in COHERE_MODELS:
            client = cohere.Client(environ["COHERE_KEY"])
            text_splitter = client
            transformer_model = model_name
        else:
            if "CUDA_VISIBLE_DEVICES" not in environ:
                raise ValueError("CUDA_VISIBLE_DEVICES is not set, you should set it since parallel inference requires GPU support")
            text_splitter = SentenceTransformersTokenTextSplitter(
                model_name=model_name,
                chunk_overlap=chunk_overlap,
                tokens_per_chunk=tokens_per_chunk,  # Use the user-provided value
            )
            transformer_model = model_name.split("/")[-1]

        ################ Quickly create the databases ################
        # 0. Initialize vector store and create a new collection
        chroma_client = PersistentClient(
            path=environ["VECTOR_SEARCH_PATH"],
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
        for filename in ['corpus.jsonl', 'queries.jsonl']:
            if filename == "queries.jsonl":
                record_type = "query"
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

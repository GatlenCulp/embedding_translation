from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import torch
import click
import os
import gc
import safetensors
import json
from torch.utils.data import DataLoader
from flask import current_app
from tqdm import tqdm, trange
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from owlergpt.utils.json_loader import JSONDataset, collate_fn
from owlergpt.modern.collection_utils import MODEL_NAMES, OPENAI_MODELS
from torch.nn.parallel import DataParallel
import torch.multiprocessing as mp
NON_OPENAI_MODELS = [m for m in MODEL_NAMES if not any (oai in m for oai in OPENAI_MODELS)]

DATASETS = [
    "arguana", # 10K
    "fiqa", # 50K
    "scidocs", # 25K
    "nfcorpus" # 5K
]

# In your main code:
# results = parallel_embed(transformers, datasets)
# Each element in results is a tuple (embeddings, ids, documents) for a specific model
# for i, (embeddings, ids, documents) in enumerate(results):
#     print(f"Model {i}:")
#     print(f"Embeddings shape: {embeddings.shape}")
#     print(f"Number of IDs: {len(ids)}")
#     print(f"Number of documents: {len(documents)}")

# XXX get working and add the openai models, launch and then once we have dataset we should be ready to create all the layers
# (stacking, etc... as necessary if so)
class SingletonInjestor:
    """
    Hardcoded singleton injector which is going to be run on a single GPU per-injector. Each dataset is run as:
    1. Everything except `Salesforce/SFR-Embedding-Mistral`
    2. `Salesforce/SFR-Embedding-Mistral`
    3. OpenAI Text Embeddings (to be run at a different time)
    """
    def __init__(
            self,
            output_dir: Path,
            device: str
    ):
        self.output_dir = output_dir
        self.chunk_size = 256
        self.chunk_overlap = 25
        self.normalize_embeddings = False
        self.batch_size = 1024 # try to be realllly fast
        self._text_splitters = [
            lambda: SentenceTransformersTokenTextSplitter(
                model_name=m,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            ) for m in tqdm(NON_OPENAI_MODELS, desc="Creating text splitters")
        ]
        # owler fork / datasets / scidocs / corpus.jsonl
        datasets_path = Path(__file__).parent.parent.parent / "datasets" / "scidocs"
        self._transformers = [
            lambda: SentenceTransformer(m, device="cpu") for m in tqdm(NON_OPENAI_MODELS, desc="Creating transformers") # fmt: skip
        ]
        self._datasets_corpus = [
            lambda: JSONDataset(
                path=datasets_path / "corpus.jsonl",
                splitter=self._text_splitters[i](),
                model_name=NON_OPENAI_MODELS[i],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                record_type="text"
            ) for i in trange(len(NON_OPENAI_MODELS), desc="Creating datasets")
        ]
        self._datasets_query = [
            lambda: JSONDataset(
                path=datasets_path / "queries.jsonl",
                splitter=self._text_splitters[i](),
                model_name=NON_OPENAI_MODELS[i],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                record_type="query"
            ) for i in trange(len(NON_OPENAI_MODELS), desc="Creating datasets")
        ]
        self.datasets_corpus: List[List[JSONDataset]] = [
            self._datasets_corpus[1:],
            [self._datasets_corpus[0]]
        ]
        self.datasets_query: List[List[JSONDataset]] = [
            self._datasets_query[1:],
            [self._datasets_query[0]]
        ]
        self.transformers: List[List[SentenceTransformer]] = [
            self._transformers[1:],
            [self._transformers[0]]
        ]
        self.transformer_names = [
            [m for m in NON_OPENAI_MODELS[1:]],
            [NON_OPENAI_MODELS[0]]
        ]
        self.device = device
        
    @staticmethod
    def __process_single_model(args: Tuple[SentenceTransformer, JSONDataset, int]) -> Tuple[torch.Tensor, List[str], List[str]]:
        transformer, dataset, device_id = args
        transformer = transformer.to(f"cuda:{device_id}")
        
        embeddings = []
        ids = []
        documents = []
        batch_size = 1024  # Adjust based on GPU memory
        
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        for batch in tqdm(dataloader, desc=f"Embedding on GPU {device_id}"):
            docs, these_ids, text_chunks = batch
            assert len(these_ids) == len(text_chunks)
            ids.extend(these_ids)
            with torch.cuda.device(device_id):
                emb = transformer.encode(text_chunks, normalize_embeddings=True, convert_to_tensor=True)
                embeddings.append(emb)
            documents.extend(docs)
        
        return torch.cat(embeddings), ids, documents

    @staticmethod
    def __parallel_embed(transformers: List[SentenceTransformer], datasets: List[JSONDataset]):
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            raise ValueError("CUDA_VISIBLE_DEVICES should not be set just to be save")
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs")
        
        # Create chunks of work based on available GPUs
        chunks = []
        for i, (transformer, dataset) in enumerate(zip(transformers, datasets)):
            device_id = i % num_gpus
            chunks.append((transformer, dataset, device_id))
        
        # Use multiprocessing to distribute work across GPUs
        with mp.Pool(num_gpus) as pool:
            results = list(pool.imap(SingletonInjestor.__process_single_model, chunks))
        
        # Return results as a list of tuples
        return results
    
    # XXX add serial vs parallel batch and then go to OH as this finishes to get the layers done
    def ingest(selfm ):
        """
        Store outputs into a folder structure like:
        /<you should name your folder's parent>
            /<you should create a folder with the dataset's name here>
                /<corpus | query>
                    /<model_name>
                        /embeddings.safetensors
                        /ids.jsonl
                        /documents.jsonl
        """
        # Each of these is a list
        total_number_of_injestions: int = sum(len(t) for t in self.transformer_names) * 2
        num_injestions_done: int = 0
        for datasets_corpus, datasets_query, transformers, transformer_names in zip(
            self.datasets_corpus,
            self.datasets_query,
            self.transformers,
            self.transformer_names
        ):
            # transformers = [t.to(self.device) for t in transformers]
            for datasets, datasets_type in [
                (datasets_corpus, "corpus"),
                (datasets_query, "query")
            ]:
                subfolder = self.output_dir / datasets_type
                subfolder.mkdir(parents=True, exist_ok=True)
                for dataset_func, transformer_func, transformer_name in zip(
                    datasets,
                    transformers,
                    transformer_names,
                ):
                    if not "mistral" in transformer_name.lower():
                        continue # XXX
                    # Only have exist within this scope to use memory efficiently
                    dataset = dataset_func()
                    transformer = transformer_func()

                    # ...
                    transformer = transformer.to(self.device)
                    num_injestions_done += 1
                    # 1. Folder structure
                    subsubfolder = subfolder / transformer_name
                    subsubfolder.mkdir(parents=True, exist_ok=False)
                    # 2. Train
                    dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)
                    embeddings: List[List[float]] = []
                    ids: List[str] = []
                    documents: List[str] = []
                    for batch in tqdm(dataloader, desc=f"Ingesting dataset ({num_injestions_done}/{total_number_of_injestions})", leave=False):
                        these_documents, these_ids, text_chunks = batch
                        assert len(these_ids) == len(text_chunks) == len(these_documents)
                        ids.extend(these_ids)
                        embeddings.append(transformer.encode(text_chunks, normalize_embeddings=self.normalize_embeddings, convert_to_tensor=True))
                        assert isinstance(embeddings[-1], torch.Tensor)
                        documents.extend(these_documents)
                        break # XXX -- hmmm wtf
                    storeme = torch.cat(embeddings, dim=0).detach().cpu()
                    assert len(storeme) == len(ids) == len(documents), f"{len(storeme)} != {len(ids)} != {len(documents)}"
                    # same stride basically...
                    safetensors.torch.save_file({"embeddings": storeme}, subsubfolder / f"embeddings.safetensors")
                    with open(subsubfolder / f"ids.jsonl", "w") as f:
                        json.dump({"ids": ids}, f)
                    with open(subsubfolder / f"documents.jsonl", "w") as f:
                        json.dump({"documents": documents}, f)
                    transformer.cpu()
                    del embeddings, storeme, ids, documents, dataset, batch
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

@current_app.cli.command("speed_injest")
@click.option("--output", "-o", type=click.Path(), default=Path(__file__).parent.parent.parent / "data" / "embeddings")
@click.option("--device", "-d", default="cuda:0")
def speed_injest(output: str, device: str):
    """
    Commands to run (probably)
    `flask speed_injest -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings/arguana -d cuda:0`
    `flask speed_injest -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings/fiqa -d cuda:1`
    `flask speed_injest -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings/scidocs -d cuda:2`
    `flask speed_injest -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings/nfcorpus -d cuda:3`
    """
    # NOTE: not parallel since it's honestly easier to just run manually one per gpu
    click.echo(f"Output: {output}")
    click.echo(f"Device: {device}")
    click.echo("========== INITIALIZING INGESTOR ==========")
    injestor = SingletonInjestor(Path(output), device)
    click.echo("========== INGESTING DATASET ==========")
    injestor.ingest()

if __name__ == "__main__":
    main()

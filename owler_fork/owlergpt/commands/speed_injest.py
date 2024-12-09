from __future__ import annotations
import sys
from pathlib import Path
_ = Path(__file__).parent.parent.parent
print("=====================> Adding to sys.path", _)
sys.path.append(_.as_posix()) # VIEW owlergpt
print("Current sys.path:", sys.path)
import time
from typing import List, Tuple, Callable
import torch
import click
import os
import gc
import torch.nn as nn
import safetensors
import json
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from owlergpt.utils.json_loader import JSONDataset, collate_fn
from owlergpt.modern.collection_utils import MODEL_NAMES, OPENAI_MODELS
import torch.multiprocessing as mp

# LOL - increase file limit cuz why not, lets not get bug
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (65535, hard))

mp.set_start_method('spawn', force=True)
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
            device: str,
            parallelize: bool = False,
            dataset_chosen: str = "scidocs"
    ):
        self.dataset_chosen = dataset_chosen
        self.parallelize = parallelize
        self.output_dir = output_dir
        self.chunk_size = 256
        self.chunk_overlap = 25
        self.normalize_embeddings = False
        self.batch_size = 1024 # try to be realllly fast
        print(f"CAN USE {len(NON_OPENAI_MODELS)} NON OPENAI MODELS")
        print("  " + "\n  ".join(NON_OPENAI_MODELS))
        self._text_splitters: List[Callable[[], SentenceTransformersTokenTextSplitter]] = [
            lambda: SentenceTransformersTokenTextSplitter(
                model_name=m,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            ) for m in tqdm(NON_OPENAI_MODELS, desc="Creating text splitters")
        ]
        # owler fork / datasets / scidocs / corpus.jsonl
        datasets_path = Path(__file__).parent.parent.parent / "datasets" / self.dataset_chosen
        self._transformers: List[Callable[[], SentenceTransformer]] = [
            lambda: SentenceTransformer(m, device="cpu") for m in tqdm(NON_OPENAI_MODELS, desc="Creating transformers") # fmt: skip
        ]
        self._datasets_corpus: List[Callable[[], JSONDataset]] = [
            lambda: JSONDataset(
                path=datasets_path / "corpus.jsonl",
                splitter=self._text_splitters[i](),
                model_name=NON_OPENAI_MODELS[i],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                record_type="text"
            ) for i in trange(len(NON_OPENAI_MODELS), desc="Creating datasets")
        ]
        self._datasets_query: List[Callable[[], JSONDataset]] = [
            lambda: JSONDataset(
                path=datasets_path / "queries.jsonl",
                splitter=self._text_splitters[i](),
                model_name=NON_OPENAI_MODELS[i],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                record_type="query"
            ) for i in trange(len(NON_OPENAI_MODELS), desc="Creating datasets")
        ]
        self.datasets_corpus: List[List[Callable[[], JSONDataset]]] = [
            self._datasets_corpus[1:],
            [self._datasets_corpus[0]]
        ]
        self.datasets_query: List[List[Callable[[], JSONDataset]]] = [
            self._datasets_query[1:],
            [self._datasets_query[0]]
        ]
        self.transformers: List[List[Callable[[], SentenceTransformer]]] = [
            self._transformers[1:],
            [self._transformers[0]]
        ]
        _names = [x.replace("/", "_") for x in NON_OPENAI_MODELS] # flatten the directory
        self.transformer_names: List[List[str]] = [
            [m for m in _names[1:]],
            [_names[0]]
        ]
        self.device = device
        
    @staticmethod # <------ carryover from `__parallel_embed`
    def h__process_single_model(args: Tuple[JSONDataset, SentenceTransformer, str, int, int, float]) -> Tuple[torch.Tensor, List[str], List[str]]:
        try:
            torch.cuda.set_per_process_memory_fraction(args[5]) 
            torch.cuda.empty_cache()

            dataset, transformer, transformer_name, device_id, batch_size = args
            transformer = transformer.to(f"cuda:{device_id}")
            
            embeddings: List[torch.Tensor] = []
            ids: List[str] = []
            documents: List[str] = []

            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True)
            for batch in tqdm(dataloader, desc=f"Embedding on GPU {device_id}"):
                docs, these_ids, text_chunks = batch
                assert len(these_ids) == len(text_chunks)
                ids.extend(these_ids)
                with torch.cuda.device(device_id):
                    emb = transformer.encode(text_chunks, normalize_embeddings=True, convert_to_tensor=True)
                    embeddings.append(emb)
                documents.extend(docs)
                # break # DEBUG - uncomment to fix bugs and iterate faster
            # Sans because we CANNOT send lambdas over the wire (pickle)
            assert isinstance(embeddings, list)
            assert all(isinstance(e, torch.Tensor) for e in embeddings)
            assert isinstance(ids, list)
            assert all(isinstance(i, str) for i in ids)
            assert isinstance(documents, list)
            assert all(isinstance(d, str) for d in documents)
            # END Sans
            embeddings_tensor = torch.cat(embeddings).detach().cpu()
        finally:
            # Explicitly close and clean up resources
            del dataloader
            transformer = transformer.cpu()
            torch.cuda.empty_cache()
        return embeddings_tensor, ids, documents

    @staticmethod # <---- otherwise multiprocessing will not work >:(
    def h__parallel_embed(
        datasets: List[JSONDataset],
        transformers: List[SentenceTransformer],
        transformer_names: List[str],
        batch_size: int
    ):
        
        mem_amt = 1 / len(datasets)
        assert (1/80) <= mem_amt <= 1.0 # everyone should get at least 1 gig
        assert len(datasets) == len(transformers) == len(transformer_names)

        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            raise ValueError("CUDA_VISIBLE_DEVICES should not be set just to be save")
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs")
        
        # Create chunks of work based on available GPUs
        chunks: List[Tuple[JSONDataset, SentenceTransformer, str, int]] = []
        for i, (transformer, dataset) in enumerate(zip(transformers, datasets)):
            device_id = i % num_gpus
            chunks.append((dataset, transformer, transformer_names[i], device_id, batch_size, mem_amt))
        
        # Use multiprocessing to distribute work across GPUs
        print("=============== bouta launch some epic parallelism ===============")
        print("  " + "\n  ".join(f"({i}) GPU {chunk[-1]}: {chunk}" for i, chunk in enumerate(chunks)))
        with mp.Pool(num_gpus) as pool:
            results = list(pool.imap(SingletonInjestor.h__process_single_model, chunks))
        
        # Return results as a list of tuples
        return results
    
    def h__save_embedding_results(
        self,
        embedding: torch.Tensor,
        ids: List[str],
        documents: List[str],
        subsubfolder: Path
    ) -> None:
        safetensors.torch.save_file({"embeddings": embedding}, subsubfolder / f"embeddings.safetensors")
        with open(subsubfolder / f"ids.jsonl", "w") as f:
            json.dump({"ids": ids}, f)
        with open(subsubfolder / f"documents.jsonl", "w") as f:
            json.dump({"documents": documents}, f)

    # TODO(Adriano) it's not clear why parallel does NOT work and it's really annoying >:(
    # (pickle problems -> too many files open -> flask problems -> segfault)
    def h__parallel_ingest_batch(
        self,
        datasets_fns: List[Callable[[], JSONDataset]],
        transformers_fns: List[Callable[[], SentenceTransformer]],
        transformer_names: List[str],
        subfolder: Path,
        # viz
        total_number_of_injestions: int, # not used for now :P
        num_injestions_done: int # not used for now :P
    ) -> None:
        datasets: List[JSONDataset] = [fn() for fn in datasets_fns]
        transformers: List[SentenceTransformer] = [fn().cpu() for fn in transformers_fns]
        embedding_objs: List[Tuple[torch.Tensor, List[str], str, List[str]]] = SingletonInjestor.h__parallel_embed(
            datasets,
            transformers,
            transformer_names,
            self.batch_size
        )
        # Sans because we CANNOT send lambdas over the wire (pickle)
        assert all(isinstance(e, JSONDataset) for e in embedding_objs)
        assert all(isinstance(t, SentenceTransformer) for t in transformers)
        assert all(isinstance(e[0], torch.Tensor) for e in embedding_objs)
        assert all(isinstance(e[1], list) for e in embedding_objs)
        assert all(isinstance(ee, str) for e in embedding_objs for ee in e[1])
        assert all(isinstance(e[2], str) for e in embedding_objs)
        assert all(isinstance(e[3], list) for e in embedding_objs)
        assert all(isinstance(ee, str) for e in embedding_objs for ee in e[3])
        assert all(isinstance(t, nn.Module) for t in transformers)
        # END Sans
        embedding_objs = [e.detach().cpu() for e in embedding_objs]
        transformers = [transformer.cpu() for transformer in transformers]
        for embedding, ids, transformer_name, documents in embedding_objs:
            # 1. Folder structure
            subsubfolder = subfolder / transformer_name
            subsubfolder.mkdir(parents=True, exist_ok=False)
            self.h__save_embedding_results(embedding, ids, documents, subsubfolder)
        
        del embedding_objs, datasets, transformers
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def h__serially_ingest_batch(
            self,
            datasets_fns: List[Callable[[], JSONDataset]],
            transformers_fns: List[Callable[[], SentenceTransformer]],
            transformer_names: List[str],
            subfolder: Path,
            # viz
            total_number_of_injestions: int,
            num_injestions_done: int
    ) -> None:
        # 251.83986496925354 => 265.43063616752625 (no)
        # transformers = [fn().to(self.device) for fn in transformers_fns] # apparently speeds up? - naw
        for dataset_func, transformer_fn, transformer_name in zip(
            datasets_fns,
            transformers_fns,
            transformer_names,
        ):
            # Only have exist within this scope to use memory efficiently
            transformer = transformer_fn()
            dataset = dataset_func()
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
                # break # DEBUG - uncomment to fix bugs and iterate faster
            storeme = torch.cat(embeddings, dim=0).detach().cpu()
            assert len(storeme) == len(ids) == len(documents), f"{len(storeme)} != {len(ids)} != {len(documents)}"
            # same stride basically...
            self.h__save_embedding_results(storeme, ids, documents, subsubfolder)
            transformer.cpu()
            del embeddings, storeme, ids, documents, dataset, batch
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def ingest(self) -> None:
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
                if self.parallelize:
                    self.h__parallel_ingest_batch(datasets, transformers, transformer_names, subfolder, total_number_of_injestions, num_injestions_done) # fmt: skip
                else:
                    self.h__serially_ingest_batch(datasets, transformers, transformer_names, subfolder, total_number_of_injestions, num_injestions_done) # fmt: skip

@click.command()
@click.option("--output", "-o", type=click.Path(), default=Path(__file__).parent.parent.parent / "data" / "embeddings")
@click.option("--device", "-d", default="cuda:0")
@click.option("--parallelize", "-p", is_flag=True, default=False)
@click.option("--dataset", "-da", default="scidocs")
def main(output: str, device: str, parallelize: bool, dataset: str):
    """
    Commands to run (probably)
    `python3 owlergpt/commands/speed_injest.py -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings/arguana -d cuda:0`
    `python3 owlergpt/commands/speed_injest.py -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings/fiqa -d cuda:1`
    `python3 owlergpt/commands/speed_injest.py -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings/scidocs -d cuda:2`
    `python3 owlergpt/commands/speed_injest.py -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings/nfcorpus -d cuda:3`
    """
    # NOTE: not parallel since it's honestly easier to just run manually one per gpu
    assert dataset in DATASETS
    start_time = time.time()
    click.echo(f"Output: {output}")
    click.echo(f"Device: {device}")
    click.echo("========== INITIALIZING INGESTOR ==========")
    injestor = SingletonInjestor(Path(output), device, parallelize, dataset)
    click.echo("========== INGESTING DATASET ==========")
    injestor.ingest()
    click.echo("========== DONE INGESTING ==========")
    print("time taken:", time.time() - start_time)

if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path


_ = Path(__file__).parent.parent.parent
print("=====================> Adding to sys.path", _)
sys.path.append(_.as_posix())  # VIEW owlergpt
print("Current sys.path:", sys.path)
import gc
import json
import os
import time

import click
import safetensors
import torch
from pydantic import BaseModel
from pydantic import Field
from sentence_transformers import SentenceTransformer

# from torch.utils.data import DataLoader
from tqdm import tqdm


# from owlergpt.utils.json_loader import JSONDataset, collate_fn # unused
# from owlergpt.modern.collection_utils import MODEL_NAMES, OPENAI_MODELS
OPENAI_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]
COHERE_MODELS = ["embed-english-v3.0"]

MODEL_NAMES = [
    "Salesforce/SFR-Embedding-Mistral",
    "WhereIsAI/UAE-Large-V1",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
    "intfloat/e5-small-v2",
    "thenlper/gte-base",
    "thenlper/gte-large",
    "thenlper/gte-small",
    "sentence-transformers/gtr-t5-base",
    "sentence-transformers/gtr-t5-large",
    "mixedbread-ai/mxbai-embed-large-v1",
    "sentence-transformers/sentence-t5-base",
    "sentence-transformers/sentence-t5-large",
    "openai/text-embedding-3-large",
    "openai/text-embedding-3-small",
]


# from owlergpt.modern.collection_utils import model2model_dimension
def model2model_dimension(model_name: str) -> int:
    """Helper: get the size of the embedding dimension vector (1D, usually something like 768-4096)."""
    # Miscellaneous (HF)
    if "/" in model_name:
        assert model_name.count("/") == 1
        model_name = model_name.split("/")[-1]
    if model_name == "SFR-Embedding-Mistral":
        return 4096
    if model_name == "UAE-Large-V1" or model_name == "mxbai-embed-large-v1":
        return 1024
    # BGE Models (HF)
    if model_name == "bge-base-en-v1.5":
        return 768
    if model_name == "bge-large-en-v1.5":
        return 1024
    if model_name == "bge-small-en-v1.5":
        return 384
    #  E5 Models (HF)
    if model_name == "e5-base-v2":
        return 768
    if model_name == "e5-large-v2":
        return 1024
    if model_name == "e5-small-v2":
        return 384
    # GTE Models (HF)
    if model_name == "gte-base":
        return 768
    if model_name == "gte-large":
        return 1024
    if model_name == "gte-small":
        return 384
    # GTR-T5 Models (HF)
    if (
        model_name == "gtr-t5-base"
        or model_name == "gtr-t5-large"
        or model_name == "sentence-t5-base"
        or model_name == "sentence-t5-large"
    ):
        return 768
    # OpenAI Models
    if model_name == "text-embedding-3-large":
        return 3072
    if model_name == "text-embedding-3-small":
        return 1536
    # NOTE: cohere may be supported in THE FUTURE
    raise ValueError(f"Unsupported model: {model_name}")


# NOTE copied from chunk_dataset.py
class Chunk(BaseModel):
    id: str = Field(
        alias="id"
    )  # unique id for the chunk (can combine doc id with index within that chunk)
    doc_id: str = Field(alias="doc_id")
    index_in_doc: int = Field(alias="index_in_doc")
    text: str = Field(alias="text")


NON_OPENAI_MODELS = [
    m for m in MODEL_NAMES if not any(oai in m for oai in OPENAI_MODELS)
]

DATASETS = [
    # (numbers are counts for documents, there may be some longer documents -> slightly more chunks)
    "arguana",  # 10K
    "fiqa",  # 50K -> 20K
    "scidocs",  # 25K -> 20K
    "nfcorpus",  # 5K
    "hotpotqa",  # 100K -> 20K
    "trec-covid",  # too much -> 20K
]


class SingletonInjestor:
    """Ingest embeddings by computing them. Do this serially. TODO do batch later."""

    def __init__(
        self,
        output_dir: Path,
        device: str,
        parallelize: bool,
        dataset_chosen: str,
    ):
        self.dataset_chosen = dataset_chosen
        self.parallelize = parallelize
        self.output_dir = output_dir
        # self.chunk_size = 256
        # self.chunk_overlap = 25
        self.chunk_size = int(os.environ.get("CHUNK_SIZE", None))
        self.chunk_overlap = int(
            os.environ.get("VECTOR_SEARCH_TEXT_SPLITTER_CHUNK_OVERLAP", None)
        )
        # self.normalize_embeddings = False
        self.normalize_embeddings = bool(
            os.environ.get("VECTOR_SEARCH_NORMALIZE_EMBEDDINGS", None)
        )
        self.batch_size = int(os.environ.get("BATCH_SIZE", None))
        self.chunk_prefix = os.environ.get("VECTOR_SEARCH_CHUNK_PREFIX", None)
        self.query_prefix = os.environ.get("VECTOR_SEARCH_QUERY_PREFIX", None)
        _NON_OPENAI_MODELS = [
            m for m in NON_OPENAI_MODELS if m != "Salesforce/SFR-Embedding-Mistral"
        ]
        print(f"CAN USE {len(_NON_OPENAI_MODELS)} NON OPENAI MODELS")
        print("  " + "\n  ".join(_NON_OPENAI_MODELS))
        chunks_path = (
            Path(__file__).parent.parent.parent / "chunks" / self.dataset_chosen
        )
        print("-------------------------------- chunks_path", chunks_path)
        self.datasets_path = chunks_path  # lol
        self.transformer_names: list[list[str]] = [[m for m in _NON_OPENAI_MODELS]]
        print("=" * 50)
        print("\n".join(f"({i}) {m}" for i, m in enumerate(_NON_OPENAI_MODELS)))
        print("=" * 50)
        self.device = device
        print("SELF DEVICE IS ", self.device)

    def h__save_embedding_results(
        self,
        embedding: torch.Tensor,
        ids: list[str],
        documents: list[str],
        chunks: list[str],
        subsubfolder: Path,
    ) -> None:
        safetensors.torch.save_file(
            {"embeddings": embedding}, subsubfolder / "embeddings.safetensors"
        )
        with open(subsubfolder / "ids.json", "w") as f:
            json.dump({"ids": ids}, f)
        with open(subsubfolder / "documents.json", "w") as f:
            json.dump({"documents": documents}, f)
        with open(subsubfolder / "chunks.json", "w") as f:
            json.dump({"chunks": chunks}, f)

    def get_jsonl_file_doctype(self, jsonl_file: Path) -> str:
        if "corpus" in jsonl_file.name and not (
            "query" in jsonl_file.name or "queries" in jsonl_file.name
        ):
            return "corpus"
        if (
            "query" in jsonl_file.name or "queries" in jsonl_file.name
        ) and "corpus" not in jsonl_file.name:
            return "queries"
        raise ValueError(f"Unknown jsonl file type: {jsonl_file}")

    def get_jsonl_file_traintype(self, jsonl_file: Path) -> str:
        if "train" in jsonl_file.name and "validation" not in jsonl_file.name:
            return "train"
        if "validation" in jsonl_file.name and "train" not in jsonl_file.name:
            return "validation"
        raise ValueError(f"Unknown jsonl file type: {jsonl_file}")

    def get_embeddings_metadatas_target_dir_paths(
        self, jsonl_file: Path, transformer_name: str
    ) -> tuple[Path, Path, Path]:
        # 1. validate the filename and parse out the types
        jsonl_doctype: str = self.get_jsonl_file_doctype(jsonl_file)
        jsonl_traintype: str = self.get_jsonl_file_traintype(jsonl_file)
        assert (
            jsonl_file.name == f"{jsonl_doctype}_{jsonl_traintype}.jsonl"
        ), f"Expected {jsonl_file.name} to be {jsonl_doctype}.{jsonl_traintype}.jsonl"
        # 2. Dataset
        assert (
            self.dataset_chosen in DATASETS
        ), f"Expected {self.dataset_chosen} to be a dataset, but not in {DATASETS}"
        assert (
            jsonl_file.parent.name == self.dataset_chosen
        ), (
            f"Expected {jsonl_file.parent.name} to be {self.dataset_chosen}"
        )  # one dataset per :P

        # 3. validate the transformer name is not in the relative path
        assert "/" not in transformer_name

        target_path = self.output_dir / transformer_name / self.dataset_chosen
        embeddings_file_path = (
            target_path / f"embeddings_{jsonl_doctype}_{jsonl_traintype}.safetensors"
        )
        metadatas_file_path = (
            target_path / f"metadatas_{jsonl_doctype}_{jsonl_traintype}.jsonl"
        )
        return embeddings_file_path, metadatas_file_path, target_path

    def ingest(self) -> None:
        """Store outputs into a folder structure like:
        /<you should name your folder's parent>
            /<you should create a folder with the dataset's name here>
                /<corpus | query>
                    /<model_name>
                        /embeddings.safetensors
                        /ids.jsonl
                        /documents.jsonl
        """
        # [DEBUG START]
        # ls = list(self.datasets_path.glob("**/*.jsonl"))
        # ls = sorted(ls)
        # print("---------------------------------------- PROCESSING JSONL FILES ----------------------------------------")
        # for l in ls:
        #     print("> " + l.as_posix())
        # return # [DEBUG END]
        # Each of these is a list
        print("[START SANS]")
        assert len(self.transformer_names) == 1  # LMAO
        jsonl_files_list = list(self.datasets_path.glob("**/*.jsonl"))
        jsonl_files_list = [
            x
            for x in jsonl_files_list
            if x.name not in ["corpus.jsonl", "queries.jsonl"]
        ]  # we don't do these again (they are in train/validation)
        assert all(
            x.name
            in [
                "corpus_train.jsonl",
                "corpus_validation.jsonl",
                "queries_train.jsonl",
                "queries_validation.jsonl",
            ]
            for x in jsonl_files_list
        )
        _parents = set(x.parent for x in jsonl_files_list)
        assert all(len(list(_parents.iterdir())) in [4, 6] for _parents in _parents)
        assert all((parent / "corpus_train.jsonl").exists() for parent in _parents)
        assert all((parent / "corpus_validation.jsonl").exists() for parent in _parents)
        assert all((parent / "queries_train.jsonl").exists() for parent in _parents)
        assert all(
            (parent / "queries_validation.jsonl").exists() for parent in _parents
        )
        print("[END SANS]")
        for transformer_names in self.transformer_names:
            assert isinstance(transformer_names, list)
            print("WILL BE LOOKING AT TRANSFORMERS:", transformer_names)
            assert self.device is not None and self.device != "cpu"  # lol
            print("CREATING TRANSFORMERS")
            transformers = [
                SentenceTransformer(
                    t, device=self.device, token=os.environ.get("HF_ACCESS_TOKEN")
                )
                for t in tqdm(transformer_names, desc="Creating transformers")
            ]
            # transformers = [None for _ in transformer_names] # [DEBUG]
            transformer_names_folderable = [
                t.replace("/", "_") for t in transformer_names
            ]
            assert len(set(transformer_names_folderable)) == len(transformer_names)
            print(
                "---------------------------------------- PROCESSING JSONL FILES ----------------------------------------"
            )
            # TODO(Adriano) parallel plz
            for transformer, transformer_name_folderable, transformer_name in zip(
                transformers,
                transformer_names_folderable,
                transformer_names,
                strict=False,
            ):
                for jsonl_file in tqdm(jsonl_files_list, desc="Processing jsonl files"):
                    embeddings_file_path, metadatas_file_path, target_dir = (
                        self.get_embeddings_metadatas_target_dir_paths(
                            jsonl_file, transformer_name_folderable
                        )
                    )
                    target_dir.mkdir(parents=True, exist_ok=True)
                    assert (
                        not embeddings_file_path.exists()
                    )  # every write should be new!
                    assert not metadatas_file_path.exists()
                    # embeddings_file_path.touch() # [DEBUG]
                    # metadatas_file_path.touch() #  [DEBUG]
                    # print("TOUCHED") # [DEBUG]
                    with open(jsonl_file) as f:
                        lines = f.readlines()
                        _chunks = [Chunk.model_validate_json(line) for line in lines]
                        chunks = [c.text for c in _chunks]
                    assert isinstance(chunks, list)
                    assert all(isinstance(c, str) for c in chunks)
                    batched_chunks: list[list[str]] = [
                        chunks[i : i + self.batch_size]
                        for i in range(0, len(chunks), self.batch_size)
                    ]
                    assert isinstance(batched_chunks, list)
                    assert all(isinstance(b, list) for b in batched_chunks)
                    assert all(isinstance(c, str) for b in batched_chunks for c in b)
                    all_embeddings: list[torch.Tensor] = []
                    for batch in tqdm(batched_chunks, desc="Embedding chunks"):
                        batch_embeddings = transformer.encode(
                            batch,
                            normalize_embeddings=self.normalize_embeddings,
                            convert_to_tensor=True,
                        )
                        assert isinstance(batch_embeddings, torch.Tensor)
                        assert len(batch_embeddings.shape) == 2
                        assert batch_embeddings.shape[0] == len(batch)
                        assert batch_embeddings.shape[1] == model2model_dimension(
                            transformer_name
                        )
                        all_embeddings.append(batch_embeddings.detach().cpu())
                    all_embeddings_pt = torch.cat(all_embeddings, dim=0)
                    assert len(all_embeddings_pt.shape) == 2
                    assert all_embeddings_pt.shape[0] == len(chunks)
                    assert all_embeddings_pt.shape[1] == model2model_dimension(
                        transformer_name
                    )
                    # 1. Save embeddings
                    safetensors.torch.save_file(
                        {"embeddings": all_embeddings_pt}, embeddings_file_path
                    )
                    # 2. Save metadatas (just copy over)
                    with open(jsonl_file) as f1:
                        with open(metadatas_file_path, "w") as f2:
                            f2.write(f1.read())
            for transformer in transformers:
                transformer.cpu()
            del transformers
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# @current_app.cli.command("ingest_hf")
@click.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=Path(__file__).parent.parent.parent / "data" / "embeddings",
)
@click.option("--device", "-d", default="cuda:0")
@click.option("--parallelize", "-p", is_flag=True, default=False)
@click.option("--dataset", "-da", default="scidocs")
def ingest_hf(output: str, device: str, parallelize: bool, dataset: str):
    """Commands to run (probably)
    `python3 owlergpt/commands/create_regular_embeddings.py -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings_huggingface/ -d cuda:0 -da arguana`
    `python3 owlergpt/commands/create_regular_embeddings.py -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings_huggingface/ -d cuda:1 -da fiqa`
    `python3 owlergpt/commands/create_regular_embeddings.py -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings_huggingface/ -d cuda:2 -da scidocs`
    `python3 owlergpt/commands/create_regular_embeddings.py -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings_huggingface/ -d cuda:3 -da nfcorpus`
    `python3 owlergpt/commands/create_regular_embeddings.py -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings_huggingface/ -d cuda:0 -da hotpotqa`
    `python3 owlergpt/commands/create_regular_embeddings.py -o /mnt/align3_drive/adrianoh/dl_final_project_embeddings_huggingface/ -d cuda:1 -da trec-covid`
    """
    assert os.environ.get("HF_ACCESS_TOKEN") is not None
    # NOTE: not parallel since it's honestly easier to just run manually one per gpu
    assert dataset in DATASETS
    start_time = time.time()
    click.echo(f"Output: {output}")
    click.echo(f"Device: {device}")
    click.echo("========== INITIALIZING INGESTOR ==========")
    injestor = SingletonInjestor(
        Path(output), device, parallelize, dataset
    )  # <--- no side-effects
    click.echo("========== INGESTING DATASET ==========")
    injestor.ingest()  # <--- side-effects
    click.echo("========== DONE INGESTING ==========")
    print("time taken:", time.time() - start_time)


if __name__ == "__main__":
    ingest_hf()

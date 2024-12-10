A standalone parallel embedding script. It will embed for all models EXCEPT the Mistral model (i.e. for the ones highlighted below) for ALL datasets in the `./chunks/` directory. Every `.jsonl` file (which usually corresponds to a corpus or whatever) gets turned into `embeddings.safetensors` and `metadatas.jsonl`. `embeddings.safetensors` has shape `(n_chunks, 1536)` and `metadatas.jsonl` has length `(n_chunks, 1)` and each is of the type we show below.
```python
from pydantic import BaseModel

class Chunk(BaseModel):
    id: str
    doc_id: str
    index_in_doc: int
    text: str

models = [
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

MODEL_DIMENSIONS = {
    # Miscellaneous (HF)
    "UAE-Large-V1": 1024,
    "mxbai-embed-large-v1": 1024,
    
    # BGE Models (HF)
    "bge-base-en-v1.5": 768,
    "bge-large-en-v1.5": 1024,
    "bge-small-en-v1.5": 384,
    
    # E5 Models (HF)
    "e5-base-v2": 768,
    "e5-large-v2": 1024,
    "e5-small-v2": 384,
    
    # GTE Models (HF)
    "gte-base": 768,
    "gte-large": 1024,
    "gte-small": 384,
    
    # GTR-T5 Models (HF)
    "gtr-t5-base": 768,
    "gtr-t5-large": 768,
    
    # Sentence T5 (HF)
    "sentence-t5-base": 768,
    "sentence-t5-large": 768,
}
```

It works as follows:
1. It has a master script that uses `subprocess` to run one instance of a child script per dataset AND model. Each child script is given parameters via the subprocess argument call that specify:
    - How much cuda memory it is allowed to use maximum
    - Which cuda device to use (it only uses one)
    - Which dataset to use to embed
    - Which model to use to embed
    - Where the chunks directory is
    - Where to store the output
2. It has a bunch of CLI child processes that get invoked and return success or failure and embeds everything into the output directory.

Note that the master script does NOT (repeat: NOT) actually use fork, spawn, multiprocessing etc... from python; it also does NOT use torch or even import torch. It simply creates a bunch of subprocesses that run the trainings. This way we never run into torch memory issues.

You call `parent.py` which is hardcoded with the right paths n shit (and it uses env etc...). Then IT calls `child.py`. You merely need to tell parent which devices via `CUDA_VISIBLE_DEVICES` and you're good to go.
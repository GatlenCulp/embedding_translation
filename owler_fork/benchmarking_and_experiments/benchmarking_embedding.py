from __future__ import annotations
"""
The point of this file was that I was trying to get my embedding process to go faster by making
better use of parallelization, batching, etc... In the end I didn't actually do it - adrianoh
"""

from typing import List
import yaml
from sentence_transformers import SentenceTransformer
import torch
from datasets import load_dataset
import time
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

class Chunker:
    def __init__(self, tokenizer: AutoTokenizer, chunk_size: int, dataset_key: str = 'text'):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size    
        self.dataset_key = dataset_key

    def chunk_text(self, text: str) -> List[List[int]]:
        """Split text into chunks of chunk_size tokens"""
        tokens: List[int] = self.tokenizer.encode(text)
        assert isinstance(tokens, list)
        assert all(isinstance(token, int) for token in tokens)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            this_chunk_size = min(self.chunk_size, len(tokens) - i)
            assert i + this_chunk_size <= len(tokens)
            chunk_tokens = tokens[i:i + this_chunk_size] + [self.tokenizer.eos_token_id for _ in range(this_chunk_size, self.chunk_size)]
            assert all(isinstance(token, int) for token in chunk_tokens), chunk_tokens
            chunks.append(chunk_tokens)
        return chunks
    
    def chunk_dataset(self, dataset: Dataset) -> List[List[int]]:
        chunks: List[List[int]] = []
        for item in tqdm(dataset[self.dataset_key], total=len(dataset[self.dataset_key]), desc="Chunking dataset"):
            chunks.extend(self.chunk_text(item))
        assert isinstance(chunks, list)
        assert all(isinstance(text, list) for text in chunks)
        assert all(isinstance(token, int) for chunk in chunks for token in chunk)
        assert all(0 < len(chunk) <= self.chunk_size for chunk in chunks)
        return chunks


def benchmark_models(chunk_size: int = 256, batch_size: int = 16284, dataset_max_size: int = 100_000):
    """
    Testing method: heavily reduce dataset size so we can try end to end and parallelism.
    """
    # Read YAML
    with open('models.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and cache dataset subset
    print("Loading dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = dataset.select(range(dataset_max_size))
    print(f"Dataset size: {len(dataset)}")

    # Filter sentence-transformers models
    print("Chunking and creating dataset!...")
    st_models = [m for m in config['models'] if m['type'] == 'sentence-transformers']
    for model_config in st_models:
        # TODO(Adriano) somehow add the memory utilization stats here... (but it has to be reliable)
        print(f"\nBenchmarking {model_config['name']}...")
        print("Creating model...")
        model_creation_start_time = time.time()
        model = SentenceTransformer(model_config['name'], device='cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        if tokenizer.eos_token_id is None:
            print("WARNING: Tokenizer does not have an eos_token_id, using pad_token_id instead")
            tokenizer.eos_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is None:
            print("WARNING: Tokenizer does not have an eos_token_id, using bos_token_id instead")
            tokenizer.eos_token_id = tokenizer.bos_token_id
        assert tokenizer.eos_token_id is not None
        model_creation_duration = time.time() - model_creation_start_time
        print(f"Model creation time: {model_creation_duration:.2f}s")
        
        # Time the embedding
        embedding_start_time = time.time()
        print("Chunking/tokenizing texts...")
        chunker = Chunker(tokenizer, chunk_size)
        tokenized_chunks: List[List[int]] = chunker.chunk_dataset(dataset)
        embeddings = model.encode(tokenized_chunks, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True)
        duration = time.time() - embedding_start_time
        total_duration = time.time() - model_creation_start_time
        
        # Memory after embedding
        mem_after = torch.cuda.memory_allocated()
        
        print(f"Peak memory during embedding: {(mem_after) / 1024**2:.2f} MB")
        print(f"Time to chunk + embed {len(tokenized_chunks)} chunks: {duration:.2f}s")
        print(f"Time to chunk + embed PER chunk: {duration/len(tokenized_chunks):.2f}s")
        print(f"Total time: {total_duration:.2f}s")
        print(f"Speed: {len(tokenized_chunks)/duration:.2f} chunks/second")
        print(f"Embeddings shape: {embeddings.shape}")
        
if __name__ == "__main__":
    benchmark_models()
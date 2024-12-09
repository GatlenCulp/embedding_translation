from flask import current_app
from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def get_openai_model_size(model_name: str) -> int:
    """ Helper. """
    if model_name.startswith("openai/"):
        model_name = model_name[len("openai/"):]
    client = OpenAI(api_key=os.environ["OPENAI_KEY"]) # <---- export properly beforehand
    list_obj = client.embeddings.create(input=["hi"], model=model_name).data[0].embedding
    assert isinstance(list_obj, list)
    assert all(isinstance(x, float) for x in list_obj)
    return len(list_obj)

def get_hf_model_size(model_name: str, device: str) -> int:
    """ Helper. """
    shape = SentenceTransformer(model_name, device=device).encode('hi').shape
    assert len(shape) == 1
    return shape[0]

@current_app.cli.command("model_sizes_info")
def model_sizes() -> None:
    """ Helper script to get the sizes of the embedding dimensions to use in our code elsewhere"""
    models = [
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
    device = os.environ["VECTOR_SEARCH_SENTENCE_TRANSFORMER_DEVICE"]
    print(f"Using device: {device}")
    openai_models = [model for model in models if model.startswith("openai/")]
    non_openai_models = [model for model in models if not model.startswith("openai/")]

    print("Getting OpenAI model sizes...")
    openai_model_sizes = [(model, get_openai_model_size(model)) for model in tqdm(openai_models, desc="OpenAI Models", total=len(openai_models))]
    print("Getting HF model sizes...")
    non_openai_model_sizes = [(model, get_hf_model_size(model, device)) for model in tqdm(non_openai_models, desc="HF Models", total=len(non_openai_models))]
    sizes = non_openai_model_sizes + openai_model_sizes
    print("="*80)
    print('\n'.join(f"{model:<40} | Shape: {shape}" for model, shape in sizes))
    print("="*80)
    print('\n'.join(("if " if i == 0 else "\nelif ")+ f"model_name == \"{model.split('/')[-1]}\":\n    return {shape}" for i, (model, shape) in enumerate(sizes)))
    print("="*80)

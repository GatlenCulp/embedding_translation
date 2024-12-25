import os

from flask import current_app
from tqdm import tqdm

from owlergpt.modern.collection_utils import MODEL_NAMES
from owlergpt.modern.collection_utils import ModelLatentSizing


@current_app.cli.command("model_sizes_info")
def model_sizes() -> None:
    """Helper script to get the sizes of the embedding dimensions to use in our code elsewhere"""
    device = os.environ["VECTOR_SEARCH_SENTENCE_TRANSFORMER_DEVICE"]
    print(f"Using device: {device}")
    openai_models = [model for model in MODEL_NAMES if model.startswith("openai/")]
    non_openai_models = [
        model for model in MODEL_NAMES if not model.startswith("openai/")
    ]

    print("Getting OpenAI model sizes...")
    openai_model_sizes = [(model, ModelLatentSizing.get_openai_model_size(model)) for model in tqdm(openai_models, desc="OpenAI Models", total=len(openai_models))]  # fmt: skip
    print("Getting HF model sizes...")
    non_openai_model_sizes = [(model, ModelLatentSizing.get_hf_model_size(model, device)) for model in tqdm(non_openai_models, desc="HF Models", total=len(non_openai_models))]  # fmt: skip
    sizes = non_openai_model_sizes + openai_model_sizes
    print("=" * 80)
    print("\n".join(f"{model:<40} | Shape: {shape}" for model, shape in sizes))
    print("=" * 80)
    print("\n".join(("if " if i == 0 else "\nelif ")+ f'model_name == "{model.split('/')[-1]}":\n    return {shape}' for i, (model, shape) in enumerate(sizes)))  # fmt: skip
    print("=" * 80)

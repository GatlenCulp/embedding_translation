# Plan

## Overview

This project explores cross-encoder training between different embedding spaces. The goal is to learn mappings between pairs of embedding models and evaluate their performance.

## Architecture Types

We will test 3 cross-encoder architectures:

1. Linear transformation
2. Single-layer MLP
3. Two-layer MLP

## Model Widths

For each architecture, we'll experiment with different widths:

- Linear: Full rank (matching input/output dimensions)
- MLPs:
  - Full rank
  - 8x Full rank hidden dimensions
  - 32x Full rank hidden dimensions

## Embedding Models

We'll use 3-4 foundation models for generating embeddings:

1. Small T5 model
2. Another small Hugging Face model
3. GPT embeddings
4. (Optional) Medium-sized model like Llama-base

## Experimental Setup

For each combination of:

- Dataset
- Embedding model pair (A, B)
- Direction (A→B and B→A)
- Architecture type and width
  Total: ~36 experiments

## Analysis Pipeline

For each experiment:

1. Train using MSE loss
2. Generate embeddings on test dataset
3. Create UMAP visualization with prompts/labels
4. Generate tables comparing:
   - MSE across model pairs and ranks
   - k-NN edit distance metrics

## Timeline

Estimated training time:

- ~1 hour per model
- 36 hours serial training
- ~9 hours with 4x parallelization

## Deliverables

1. Training pipeline implementation
2. Visualization code and blog framework
3. Initial blog with placeholder data
4. Final blog with real experimental results

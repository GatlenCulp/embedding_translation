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



---

Notes:

ChromaDB

(Embeddings x Number Embeddings) dka difference?
(Layer?) MSE plot error vs dimension

You get an embedding dataset for each (Text Dataset x Model for Embedding x Translation Model)

Training/Test split

HotPotQA for TextData. (Queries and Prompts, Every dataset has a list of texts. Queries the user asks.)

They have a similarity analysis

Unit of comparison is an embedding dataset

KNNs within a query dataset. Dissimilar. Something something.

CKA metric. Dot product/euclidean distance of kernel matrix (ie: take a dataset and take similarities between two to see if matrix is the same)

<!-- https://www.google.com/search?sca_esv=b6c726d8a6b78b99&sxsrf=ADLYWIJuJaaqMdFx8nN3yKnCbe4yLa7eUA:1733711783659&q=cka+matrix+kernel&source=lnms&fbs=AEQNm0DvD4UMlvdpwktgGj2ZHhIXi58ra1MCmgeKkVE8y_uPCA_VArK8eDJ3eXUe-YWeaCBH-amb2Pf6-dxDrKVpML-nj-Q0usp9oUfQBWygHdzlPQyK7ekinri2xZNEevPNnSWXBwllpEgX7aD0z7Bz9Smf6FJNeCdJxssgOmj0CiTHs___g-FoVbY408ufrjMLdYcAMcnx3YBi6Ge-Y81wpmFeMjBDzQ&sa=X&ved=2ahUKEwi674fY05mKAxVvAHkGHQfjDmgQ0pQJegQIFBAB&biw=1742&bih=1028&dpr=2 -->

Jacard index. Rank based harmonic something.

<!-- Outline -->

Introduction of what we did.

Architecture diagram for each of our models + showing what we measure.

Kernel Matrices are/are not different. Some examples of searches. Which documents were fetched in both.

80-20 split, do 20% semantic search and compare.

CKA Quantitative aggregate
Ranking similarity function something

Conclude this a bit more general.

Qualitative models. Aggregate metrics.

Stitch from one latent space to another. Relationships more in new model vs old model.

---

ChromaDB dataset

For every embedding translator, what is from and to. Save into ChromaDB.

Aggregate statics, MSE across permutation

1. One dataset
2. Two models
3. Two embedding translators with four analyses (A -> B) (B -> A)
4. Kernel Tables
5. Aggregate Statistics, MSE (Across all permutations) and KNN (Did the 20%)


----

## Updated MVP Plan

Some notation:
1. $D$ textual datasets (possibly HotPotQA) [$D = 1$]
2. $C$ cross-encoders [$C = 2$]
3. $Z$ embedding models
4. $z_\texttt{A}$ denotes an embedding model named $\texttt{A}$
5. $\varepsilon_\texttt{A}$ denotes the true embedding space of an embedding model named $\texttt{A}$
6. $c^{\texttt{(architecture)}}_{\varepsilon_\texttt{A} \to \hat{\varepsilon}_\texttt{B}}$ denotes an embedding translator with $\texttt{architecture}$ trained to translate embeddings such that $c^{\texttt{(architecture)}}_{\varepsilon_\texttt{A} \to \hat{\varepsilon}_\texttt{B}}(z_\texttt{A}(x)) \approx z_\texttt{B}(x)$
7. There are $C$ total embedding translators
8. $\hat{\varepsilon}_\texttt{A}$ denotes an approximate embedding space of an embedding model named $\texttt{A}$ (there may be multiple)
9. $E$ embedding datasets [$E = X * D$]
10. $\hat{E}_{\varepsilon_\texttt{A} \to \hat{\varepsilon}_\texttt{B}}$ are translated embedding datasets. bah
11. Lets say there are $A$ architectures

\\

1. Load $D$ textual datasets (possibly HotPotQA) [$D = 1$]
2. Split each dataset $d \in \{1, \cdots, D\}$ datasets into 80% $d_\texttt{train}$, 20% $d_\texttt{test}$
3. For each of $Z$ embedding models and each of $D$ datasets, create $E = Z * D$ embedding datasets.
4. For each of $A$ architectures and $\forall i, j \in \{1, ..., E\}, i \neq j$, train embedding translator $c^{A}_{i \to j}$ creating $A * E * (E - 1)$ translation embedding datasets
5. Aaaaaa
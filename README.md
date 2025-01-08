# LEAD: Linear Embedding Alignment across Deep Neural Network Language Models’ Representations

_Note: This project/paper is not yet fully complete._

This is a repository containing the source code for the [LEAD Blog Post](https://gatlenculp.github.io/embedding_translation/) and Gatlen Culp and Adriano Hernandez from MIT.

Published December 10th, 2024

## Abstract

Recent advances in Large Language Models (LLMs) have demonstrated their remarkable ability to capture semantic information. We investigate whether different language embedding models learn similar semantic representations despite variations in architecture, training data, and initialization. While previous work explored model similarity through top-k results and Centered Kernel Alignment (CKA), yielding mixed results, in the field of large language embedding models, which we focus on, there is a gap: more modern similarity quantifiation methods from Computer Vision, such as model stitching, which operationalizes the notion of “similarity” in a way that emphasizes downstream utility, are not explored. We apply stitching by training linear and nonlinear (MLP) mappings, called “stitches” between embedding spaces, which aim to biject between embeddings of the same datapoints. We define two spaces as connectivity-aligned if stitches achieve low mean squared error, indicating approximate bijectivity.

Our analysis spans 6 embedding datasets (5,000-20,000 documents), 18 models (between 20-30 layers, including both open-source and OpenAI models), and stitches ranging from linear stitches to MLPs 7 layers deep, with a focus on linear stitches. We hoped that stitching would recover the similarity between models, aligning with a strong interpretation of the platonic representation hypothesis. However, things appear to be more complicated. Our results suggest that embedding models are not linearly connectivity-aligned. Specifically, linear stitches do not perform significantly better than mean estimators. A brief foray into MLPs suggests that training shallow MLPs does not necessarily work out of the box either, but more work remains to be done on non-linear stitches, since we haven’t fully maximized their potential here. Stitches are important, because their success can be used to determine operational, and therefore useful, notions of representational similarity. Our findings buttress the hypothesis that alignment metrics such as CKA are not always informative of behavior or feature overlap between models.

To read the rest of blog, click [here](https://gatlenculp.github.io/embedding_translation/).

## Other Resources

View our HuggingFace datasets [here](https://huggingface.co/datasets/GatlenCulp/LEAD-Embeddings) and our trained stitches [here](https://huggingface.co/GatlenCulp/LEAD-Stitch-Models).
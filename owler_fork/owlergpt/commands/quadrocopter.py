from __future__ import annotations
"""
TODO(Adriano) finish this (it might turn into a jupyter notebook idk)

This is an offline script to basically do the same thing as `quadracopter.ipynb`. It will provide functionality (through
multiple click commands) to:
1. Train optimal OLS and batched OLS for all pairs of embedding spaces, speeding up acquisition of our results by 1000x
2. Be able to calculate a set of visualizations that we need for the paper and store them somewhere
    - MAE Table
    - MSE Table
    - Variance Explained Table
    - MAE Explained Table
    - Aggregated of the above (using harmonic/arithmetic means)
    - CKA, Ranking Jaccard Similarity, and Rank score (to compare with the previous tables)
        - Cluster for CKA like the old paper
        - Sample a few random pairs of models and dataset and showcase the triplet loss
        - Cherry pick rank similarity from one model to all others and same for Jaccard like in the paper (this is done both in the
            bofore stitch section and in the after stitch section)
        - Table of rank similarity and jaccard similarity over top 10
    - Kernel maps
        - Sample a few random pairs of models and dataset and showcase the triplet loss
    - UMAP, PCA, and t-SNE for embeddings (combined with a clustering algorithm to get some labels on this shit)
        - Bonus: find some relevant labels for the chunks and then add that in there?
    - MSE/MAE of single cycle
    - Randomly sample some different transformations and calculate over-cycle performance (i.e. so we can see how long
        it takes to get error up by some percent like a half-life -> geometric mean)
    - Random matrix controls for CKA, Jaccard, MAE, MSE, etc... with sampling to provide error bars
(parses outputs from OLS or GD outputs and then creates this shit)
3. 
"""

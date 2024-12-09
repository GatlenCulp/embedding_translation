# Plan
Ok here is the proposed V1 of this shit (answer two questions: (1) does platonic rep hypothesis apply to embedding search models for text, (2) are they linearly connected/how easy are they to connect?):
1. Pick 5 datasets - DONE
2. Pick 5 commonly used embedding models - DONE
3. Embed each for each dataset (25 combinations)
    - Be able to embed one model on one dataset (test)
    - Be able tos embed one model for all datasets (test)
    - Be able to batch across model(s) and datasets (test)
    - Run the monitor the final inference
4. (do this after 5 for pipelining) Between each pair of embeddings measure the alignment using whatever alignment methods we have that do not require any more training. This answers the first question. (include not only CKA, kernel tables, kernel tables per 5 known features features incl 3 non-abstract and 2 abstract (noting that which features may vary by dataset) ranking edit distance @ topk, IoU @ topk, rank @ topk (from https://arxiv.org/pdf/2407.08275v1, did not fully understand but can use), pca/umap/tsne visualization, and whatever is available in the platonic rep. hypothesis paper/library---consider PRing new such difference methods to the library as a way to build some clout!)
5. On each dataset, per pair of embedding models, train a stitch both ways (try: permutation, orthogonal transformation, mean-centered orthogonal affine transformation, same but learned mean, linear, linear affine, 1-layer MLP) and for each stitch measure (1) MSE/MAE accuracy, (2) edit distance between pre-stitch and post-stitch both ways, (3) visualize umap/tsne/pca and compare visually, (4) run a clustering algorithm from the same initialization and compare the results.
6. Write up the blog, imitate https://deep-learning-mit.github.io/staging/blog; include also a ballpark estimate of costs simply based on the proportion of the embedding model size that the stitch size.

According to the embeddings paper it seems that CKA does not necessarily correlate with behavioral similarity. A natural question is whether some other metric might. Not sure what a better hypothesis might actually look like but I think for now the plan does not change.

Questions on what we might vary and how much should we:
- Chunking strategy and size?
- The paper's (https://arxiv.org/pdf/2407.08275v1) ranking comparison methods are not always ideal since they might not take into account the fact that there may be different variances. I'm also curious if we can convert one model's ranking into another's without training or generally training for very little


## What is important to do well AFAIK
1. Clear communication
    - Hypothesis should be made very explicit and clear () -> we should have more than 1 hypothesis
    - Conclusion should also be ^
    - There should be good images and a


## Mistakes made in proposal
Length matters

Perhaps also state how you plan to assess 'complicated' translations from not. 

Planned set of experiments apparently not clear

Each claim you make should be precise and supported. You can support a claim either bya citation, an experiment, or a mathematical argument.

technical writing, not overly formal nor overly informal

further our understanding ofdeep learning

Your blog should be around 2000–3000 words and contain 4–6 figures/tables visualizing aspects of your study (i.e. it should be slightly shorter than Karpathy's blogs, Lil Weng/Log's blogs, https://www.engraved.blog/why-machine-learning-algorithms-are-hard-to-tune, etc...)

do not go too far above the word count (i.e. maybe keep it under 6000 words)

1.5 depth of content of a single person: the number of experiments you run orthe number of hypotheses you investigate

graded by novelty, quality, and clarity of the content: grade will be determined by how well your blog offers fresh insights and in-depth analysis.
    - QUESTION: IS OUR IDEA FRESH/NEW ENOUGH? WHAT ARE SOME BIG QUESTIONS WE COULD TRY AND ANSWER?

idea of eigenvalue/singular value sizes as denoting difference (or generally, if there is some sort of dimensional ordering by singular value permutation----i.e. of PCA)

idea to look at SAE or splice features

reducing the number of calls might be good: https://docs.google.com/spreadsheets/d/14ozzIYPRjoPyciJGNdPMCa8GlhY7vu1iUL1HdzYi0ts/edit?gid=0#gid=0 (look for steering vectors for future queries, we can train a model that will PICK one of our steering vectors and then diff it or something like that)

question: do different models share the same biases? this is something we can do with this research...

Questions we can answer:
1. Do different embeddings change search rankings and if so in what ways and how much? Could that depend on the dataset the model was trained on (i.e. is this probabilistically heavily dependent on the dataset it was trained on)?
2. Do different embedding models learn different relationships between things?
3. Do different embeddings exhibit any similar or different biases? Again, is this based on the dataset?
4. Can you train a "diff" model or MoE to just add a small diff to a steering vector at a much cheaper cost?
5. How much can you reduce the cost of re-calculating embeddings in a new space given the embeddings in an old space (i.e. when a new embedding model comes out)? How much CO2 do you save? How much $$$ do you save?
6. What is the scale of agreement with human annotation as we scale the size of the embedding model? Do models get bigger at a steady rate or sort of plateau early on?
7. Are embeddings well-approximated by bag of words for a lot of text or vision models? Does that differ by modality?
8. How much noise can we add to embeddings before the model flips its ranking/prediction?
9. If we train an adversarial model to perturb embeddings under a perturbation budget, how low can the budget go to get complete deletion of the embedding?
10. 

# Feedback from Sharut Gupta
"maybe do a little more literature research, but linear transformability would be very useful and good to know and a great contribution"

"read some newer papers to know what's actually going on right now" (she doesn't seem to know the details of this field tbh)

NOTE TO SELF THERE IS A LOT OF SHIT HERE AND WE NEED TO ACTUALLY REDUCE TO GET SOMETHING DONE TO START WITH!

THIS IS TOO MUCH

should train some ses or something?

## TODO immediately (actually not really ngl)
1. go through the list of papers and extract the relevant ones
- https://phillipi.github.io/prh/ (recommended)
- https://arxiv.org/abs/2407.08275v1 (RAG, might suck but is the setting we are looking at)
- https://arxiv.org/abs/2310.13018 (really good overview, including also neuroscience research)

# Things we should read
- TODO something about SAEs should be cited and noted
- TODO Look through this workshop: https://representational-alignment.github.io/
- TODO https://openreview.net/forum?id=fLIWMnZ9ij (did not fully understand, but it seems like they are using SVD and look at alignment of the principal components/vectors)
- TODO Look at who cited me and who I cite in my original model stitching paper
- TODO https://uchicago.app.box.com/s/ymyt6ushjg94l1aauutgwq256w30mhbg (CKA slides about methods for measuring alignment)
- TODO Find out how to measure search algorithm performance differences (i.e. other than edit distance)
- TODO https://openreview.net/forum?id=YY2iA0hfia (seems like they are trying to mechanistically link network data encoding (representations) to function (i.e. whether those representations are actually used for the downstream prediction); this is different from what we are doing but is actually worth reading because we might be curious to understand _why_ the rankings change and we might wnat to take away the ablationary analysis)
- https://arxiv.org/abs/2402.10376 (SpLiCE; I roughly know what this is and  how it works, but it may be useful to understand later)
- https://arxiv.org/abs/2403.05440 (normalized cosine similarity is not always best; I don't think I need to read too much into this, my measures should take into account both normalization and no normalization)

# Other research but not worth reading (yet)
- https://arxiv.org/abs/2301.11990 (in CV representational alignment with humans is useful)
- https://openreview.net/forum?id=7STegP98cT (they find that representations are not the same as humans and might not be about concepts among other things in the setting of a game)
- https://arxiv.org/pdf/2410.06940 (recommended, but doesn't seem _immediately_ relevant; the key idea here seems to be that diffusion model training can be improved and sped up by aligning representations with that of a pretrained contrastively-trained image encoder (i.e. it seems like the latent or even images are projected onto the space for the contrastive method and then dot product'ed or something like that))
- https://www.semanticscholar.org/paper/Revisiting-Embedding-Based-Entity-Alignment%3A-A-and-Sun-Hu/5ae9526438d7b029518e6d87c51a4daebbb5225f (seems to do entity alignment in some sort of "fusion" of multiple embeddings/sensors/etc...)
- https://www.semanticscholar.org/paper/Exploring-Alignment-of-Representations-with-Human-Nanda-Majumdar/37ed8a693b396d05c1a3bbbea0683b3b899933e7 (these guys propose a way of measuring alignment w.r.t. humans and probably find that alignment with human perceptions are good)
- https://arxiv.org/abs/2305.19294 (they introduce a method LIKE CKA but for SINGLE DATAPOINTS so that they can see how those single datapoints have their representations change over time)
- https://aclanthology.org/2020.acl-main.422.pdf (old paper that compares different architectures for representation similarity and finds that they are more similar than expected)

# Things observed in the research
- A preponderance of focus on _vision_ and very little on _NLP_ (this can be one way we can start to make an impact)

# Distill template
(it seems to kind of just work OK, so I might just use the distill template)
- https://github.com/distillpub/template
- https://distill.pub/guide/

They use distill so we should too.
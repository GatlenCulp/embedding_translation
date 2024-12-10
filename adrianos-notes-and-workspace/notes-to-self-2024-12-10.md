# Things going on generally important
(low priority) Missing mistral on nfcorpus OOM
    - Solution 1: ignore this model
    - Solution 2: re-embed after we fix higher priority things
(high priority) A bunch of embeddings DONT HAVE MATCHING CHUNKS because owler chunks based on the tokenizer num tokens which may not match (fuck these guys)
    - Solution 1: Find out which models are mappable to which models and then just train layers between all those pairs right now
    - Solution 2: chunk using the GPT splitter, then tokenize using the actual splitter or tokenizer or whatever so we
        have consistency at least with OpenAI embeddings. Then re-chunk-and-embed all of [arguana, fiqa, scidocs, nfcorpus]
(ultra high priority) Have a layer's embedding dataset (at least a single pair of models on a single dataset)
(high priority) Create smaller datasets randomly sampled from each of the datasets so that we can generally just re-embed (do solution 2 from above for each of these guys) much much faster. It should take at most 5 minutes to embed and we don't need more than around 1000 randomly sampled (diverse) exampls per document.
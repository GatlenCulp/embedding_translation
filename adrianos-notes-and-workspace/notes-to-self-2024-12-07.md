# Command template
```bash
conda activate python311 && cd /mnt/align3_drive/adrianoh/git/embedding-model-similarity_no_fork/ && export $(cat .env | xargs) && export VECTOR_SEARCH_SENTENCE_TRANSFORMER_MODEL="sentence-transformers/gtr-t5-large" && export CUDA_VISIBLE_DEVICES=5 && export VECTOR_SEARCH_SENTENCE_TRANSFORMER_DEVICE="cuda" && flask ingest_ds
```


az storage blob upload-batch --account-name ifyoudontpremium --source 

Missed `arguana_256` and `scidocs_256` with model `Salesforce/SFR-Embedding-Mistral` and all thereafter (because OOM with Cuda) THESE ARE NOW DONE.

Missed `nfcorpus_256` with model `text-embedding-3-small` and all thereafter because of an authentication error (probably will need to redo OpenAI embeddings for _all_ datasets and I will need to set up cohere to get the full analysis, but for now I think it's fair to ignore cohere; note that arguana and scidocs may be finished OK as well since they will be using the new key). THIS IS NOW DONE.

<!-- - restarted trec covid for mistral and new openai key
- missing shit
    -`fiqa_256` starting on `Salesforce/SFR-Embedding-Mistral` (also `mxbai`, `uae` and `openai`)
    - Up to `intfloat/e5-small-v2` on `hotpotqa_256`, but it's not clear if I've done the ones thereafter or not?
    - `hotpotqa_256` for `sentence-transformers/gtr-t5-large` still ongoing
    - `trec-covid` is alost done with everything (i.e. text embeddings)
    - TODO we should try and parallelize fiqa for text embeddings as well as see if hotpotqa is done or whats the deal there generally -->

So by model:
- fiqa is working on everything after salesforce
- hotpotqa seems to be working on the initial gtr and everything AFTER e5 small
- AFAIK trec-covid is done
- AFAIK nfcorpus is done
- AFAIK scifacts is done
- AFAIK arguana is done


Problems with this system:
- Doesn't do batch (so I need to invoke it multiple times)
- Not parallelized properly (i.e. across GPUs and on the same GPU), so it's really slow
- Doesn't do some sort of sanity test to make sure that 
- Doesn't use hogger
<!-- - Chroma might be slower than FAISS? This doesn't matter because we don't actually care about searching so much, we primarily want to be able to very quickly embed a lot of text. -->
- Index search might not be using GPU? WTF?

How we query might really matter for comparing the top-k results and Jaccard similarity:`https://community.openai.com/t/should-i-modify-user-queries-before-semantic-search/393047`. We might also look for invariance to prompts! This is looking complicated >:(
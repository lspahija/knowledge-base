# Knowledge Base

### Run

Run with `python -m uvicorn app:app --reload`

### Index

A queryable in-memory index is created over the files in the `data` directory upon startup.

New documents can be added to the index during runtime by sending a POST to the `/document` endpoint.

The default index used is the in-memory `GPTSimpleVectorIndex`. The vector db Pinecone is also supported.
To use Pinecone, set the `INDEX_TYPE` env var to `pinecone` and populate the `PINECONE_API_KEY` env var.

### LLM

The default LLM used is `flan-t5-large`, an Apache-2.0-licensed LLM that is downloaded once upon startup and then runs
locally.

To instead use `gpt-3.5-turbo`, set the `LLM_MODEL_NAME` env var to `gpt-3.5-turbo`.
If `gpt-3.5-turbo` is used, the `OPENAI_API_KEY` env var must also be set.

### Sample Requests

`main.http` contains sample HTTP requests for the REST API.



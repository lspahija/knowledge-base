import os
import time

import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LLMPredictor, LangchainEmbedding, ServiceContext, GPTSimpleVectorIndex, \
    SimpleDirectoryReader, Document, GPTPineconeIndex
from llama_index.indices.base import BaseGPTIndex

from llm import get_llm

INDEX_TYPE = os.getenv("INDEX_TYPE", "in-memory")


def create_index() -> BaseGPTIndex:
    documents = SimpleDirectoryReader('data').load_data()

    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(llm=get_llm()),
        embed_model=LangchainEmbedding(HuggingFaceEmbeddings()))

    match INDEX_TYPE:
        case "in-memory":
            return GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
        case "pinecone":
            return create_pinecone_index(documents)
        case _:
            raise Exception(f"unsupported index type: {INDEX_TYPE}")


def create_pinecone_index(documents: list[Document]) -> BaseGPTIndex:
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
    index_name = "knowledge-base"

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=1536, metric="cosine", pod_type="p1")
        wait_on_pinecone_index(index_name)

    return GPTPineconeIndex.from_documents(documents, pinecone_index=pinecone.Index(index_name))


def wait_on_pinecone_index(index: str):
    while True:
        try:
            desc = pinecone.describe_index(index)
            if desc[7]['ready']:
                return
        except pinecone.NotFoundException:
            pass

        print("pinecone index not yet ready. checking again in 5 seconds...")
        time.sleep(5)

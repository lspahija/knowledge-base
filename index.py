import os
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

    return get_configured_index(service_context, documents)


def get_configured_index(service_context: ServiceContext, documents: list[Document]) -> BaseGPTIndex:
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
        pinecone.create_index(name=index_name, dimension=768, metric="cosine", pod_type="p1")

    return GPTPineconeIndex.from_documents(documents, pinecone_index=pinecone.Index(index_name))

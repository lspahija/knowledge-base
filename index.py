from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LLMPredictor, LangchainEmbedding, Document, ServiceContext, GPTSimpleVectorIndex, \
    SimpleDirectoryReader
from llama_index.indices.base import BaseGPTIndex


def create_index() -> BaseGPTIndex:
    documents = SimpleDirectoryReader('data').load_data()

    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")),
        embed_model=LangchainEmbedding(HuggingFaceEmbeddings()))

    return GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

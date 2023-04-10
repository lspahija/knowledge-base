import os
from transformers import pipeline
from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI

from langchain.schema import BaseLanguageModel

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "flan-t5-large")


def get_llm() -> BaseLanguageModel:  # https://news.ycombinator.com/item?id=35512338
    match LLM_MODEL_NAME:
        case "flan-t5-large":
            return FlanT5()
        case "gpt-3.5-turbo":
            return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        case _:
            raise Exception(f"unsupported LLM model: {LLM_MODEL_NAME}")


class FlanT5(LLM):
    model_name = "google/flan-t5-large"
    pipeline = pipeline("text2text-generation", model=model_name, device="mps")

    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=9999)[0]["generated_text"]

    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "flan"

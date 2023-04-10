from pydantic import BaseModel


class DocumentRequest(BaseModel):
    document: str


class QueryRequest(BaseModel):
    query: str

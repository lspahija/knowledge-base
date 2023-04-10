from fastapi import FastAPI
from llama_index import Document

from index import create_index
from model import DocumentRequest, QueryRequest

app = FastAPI()
index = create_index()


@app.post("/document")
async def add_document_to_index(req: DocumentRequest) -> str:
    doc = Document(req.document)
    index.insert(doc)
    return doc.get_doc_id()


@app.delete("/document/{document_id}")
async def delete_document(document_id: str):
    index.delete(document_id)


@app.post("/query")
async def query_index(req: QueryRequest) -> str:
    return (await index.aquery(req.query)).response

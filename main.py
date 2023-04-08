from fastapi import FastAPI
from pydantic import BaseModel

from index import create_index

app = FastAPI()
index = create_index()


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def query_index(req: QueryRequest) -> str:
    return index.query(req.query).response

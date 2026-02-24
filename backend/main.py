from fastapi import FastAPI
from pydantic import BaseModel
from backend.retrieval import search_patents
from backend.gemini import rerank

app = FastAPI()


class Query(BaseModel):
    claim: str


@app.post("/search")
def search(query: Query):

    retrieved = search_patents(
        query.claim
    )

    reranked = rerank(
        query.claim,
        retrieved
    )

    return {
        "retrieved": retrieved,
        "reranked": reranked
    }
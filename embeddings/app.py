from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from embeddings.search_index import (
    rewrite_query,
    hybrid_search,
    answer_question,
    chunks
)

app = FastAPI()   # ← THIS WAS MISSING

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_llm(q: Query):
    rewritten = rewrite_query(q.question)
    ids = hybrid_search(rewritten, k=10)
    context = "\n\n".join([chunks[i] for i in ids])
    answer = answer_question(context, q.question)
    return {
        "question": q.question,
        "answer": answer
    }

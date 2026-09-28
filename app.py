# API server
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.graph import graph

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: list[str]
    confidence_score: float


@app.get("/")
def serve_ui():
    return FileResponse("ui/index.html")


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        res = graph.invoke({"question": req.query})
        return ChatResponse(
            answer=res["answer"],
            retrieved_chunks=res["context"],
            confidence_score=res["score"],
        )
    except Exception as e:
        print(f"Error calling graph: {e}")
        raise HTTPException(status_code=500, detail="Upstream AI error (rate limit or outage)")

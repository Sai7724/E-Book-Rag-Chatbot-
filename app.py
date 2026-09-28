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
        # Send back the real error. Gemini keeps its HTTP code on e.__cause__.code,
        # Pinecone on e.status; anything else is a 500 from our side.
        print(f"Error calling graph: {e}")
        status = getattr(e.__cause__, "code", None) or getattr(e, "status", None)
        if not (isinstance(status, int) and 400 <= status < 600):
            status = 500
        raise HTTPException(status_code=status, detail=f"{type(e).__name__}: {e}")

# Agentic AI eBook RAG chatbot

A simple RAG chatbot for the Agentic AI eBook. Uses Pinecone for the vector DB, LangGraph for the RAG flow, and FastAPI for the backend.

## Project structure

```
├── data/Ebook-Agentic-AI.pdf   # the source eBook (downloaded by ingestion)
├── src/
│   ├── config.py               # env vars and constants
│   ├── ingestion.py            # PDF -> chunks -> embeddings -> Pinecone
│   └── graph.py                # LangGraph workflow (retrieve -> generate)
├── ui/index.html               # React chat UI (served at /)
├── app.py                      # FastAPI app, POST /chat + UI
├── tests_sample_queries.py     # 6 benchmark questions
├── requirements.txt
└── .env.example
```

## Setup

Requires Python 3.10+, a Gemini API key, and a Pinecone account.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then put your real keys in .env
```

## 1. Ingest the eBook (run once)

```bash
python -m src.ingestion
```

This will download the PDF to `data/` and create the Pinecone index if it doesn't exist. It splits the text into chunks and uploads them.

Note: Gemini's free tier has rate limits, so this script batches uploads and pauses for 60 seconds between batches.

If the automatic download fails, you can grab the PDF from [Google Drive](https://drive.google.com/file/d/15VLphKcY23_fpYxN62UEQRri_psRVfP9/view) and save it as `data/Ebook-Agentic-AI.pdf`.

## 2. Start the API

```bash
uvicorn app:app --reload
```

The chat UI is available at http://127.0.0.1:8000. 

Swagger docs are at http://127.0.0.1:8000/docs. Example request:

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"query": "What is Agentic AI?"}'
```

Response:

```json
{
  "answer": "...",
  "retrieved_chunks": ["...", "..."],
  "confidence_score": 0.62
}
```

## 3. Run the test queries

With the server running, in a second terminal:

```bash
python tests_sample_queries.py
```

The script asks 4 questions the eBook covers and 2 it doesn't (the 2022 FIFA World
Cup winner and the Bitcoin price). It fails if the bot answers either of the last
two.

## Architecture

```
Ingestion:
PDF -> PyPDFLoader -> TextSplitter -> Gemini embeddings -> Pinecone

Query:
POST /chat -> [retrieve] -> [generate] -> JSON
```

The graph has two nodes:
1. `retrieve`: Embeds the question and fetches the top 8 closest chunks from Pinecone. 
2. `generate`: Uses the retrieved chunks in a prompt. If the chunks aren't relevant, the model refuses to answer.

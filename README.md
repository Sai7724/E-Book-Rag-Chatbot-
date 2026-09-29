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

## Tech stack

| Part | Tool |
|---|---|
| PDF loading | `PyPDFLoader` (LangChain) |
| Chunking | `RecursiveCharacterTextSplitter`, 1000 characters, 200 overlap |
| Embeddings | Google `gemini-embedding-001`, 1536 dimensions |
| Vector store | Pinecone serverless index, 1536 dimensions, cosine metric |
| Workflow | LangGraph `StateGraph` (retrieve, then generate) |
| LLM | Google Gemini Flash models, with fallbacks |
| API | FastAPI + Uvicorn |
| UI | One HTML page served by FastAPI |
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

| # | Question | Expected result |
|---|---|---|
| 1 | What is Agentic AI according to the eBook? | Answer from the eBook |
| 2 | How do AI agents differ from traditional automation systems? | Answer from the eBook |
| 3 | What are the core components of an Agentic Architecture? | Answer from the eBook |
| 4 | What role does memory play in Agentic AI workflows? | Answer from the eBook |
| 5 | Who won the 2022 FIFA World Cup? | Refusal, score 0 |
| 6 | What is the current price of Bitcoin? | Refusal, score 0 |

For questions 5 and 6 the bot has to reply "I cannot answer based on the provided
document." The script fails if it answers either one.

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
```

### 7. Add at the end
```
## Differences from the assignment spec

- Gemini instead of OpenAI for embeddings and answers, because of API credit
  limits. `gemini-embedding-001` is used at 1536 dimensions, the same size as
  `text-embedding-3-small`, so the index matches the spec (1536, cosine).
- `langchain-pinecone` instead of `pinecone-client`, which is deprecated and
  conflicts with the newer Pinecone package.
- A small HTML chat page instead of Streamlit (Streamlit was optional).

## Known limitations

- Free-tier Gemini models are sometimes overloaded, so a request can take 10 to
  20 seconds or fail with a 503. Trying again usually works.
- The confidence score shows how close the retrieved text is to the question.
  It is not a calibrated probability that the answer is correct.
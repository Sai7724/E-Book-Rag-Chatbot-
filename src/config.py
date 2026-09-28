import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
PDF_PATH = "data/Ebook-Agentic-AI.pdf"
PDF_URL = "https://drive.google.com/uc?export=download&id=15VLphKcY23_fpYxN62UEQRri_psRVfP9"

EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768

# fallback models just in case
LLM_MODELS = ["gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.8-flash"]
TOP_K = 8

REFUSAL = "I cannot answer based on the provided document."

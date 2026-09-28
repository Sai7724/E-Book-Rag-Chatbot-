import os
import time
import urllib.request

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src.config import EMBED_DIM, EMBED_MODEL, PDF_PATH, PDF_URL, PINECONE_INDEX_NAME

BATCH_SIZE = 90  # Gemini's free limit is 100 per minute


def download_pdf():
    if not os.path.exists(PDF_PATH):
        os.makedirs("data", exist_ok=True)
        print("Downloading PDF...")
        urllib.request.urlretrieve(PDF_URL, PDF_PATH)


def create_index():
    pc = Pinecone()
    existing_indexes = pc.list_indexes().names()
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating index {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def run_ingestion():
    download_pdf()
    create_index()

    print("Loading pages...")
    pages = PyPDFLoader(PDF_PATH).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, output_dimensionality=EMBED_DIM)
    store = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    ids = [f"chunk-{i}" for i in range(len(chunks))]

    # need to batch this because of rate limits
    for i in range(0, len(chunks), BATCH_SIZE):
        if i > 0:
            print("Sleeping for rate limits...")
            time.sleep(60)
            
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]
        
        store.add_documents(batch_chunks, ids=batch_ids)
        print(f"Uploaded up to chunk {i + len(batch_chunks)} of {len(chunks)}")
        
    print(f"Finished ingesting {len(chunks)} chunks!")


if __name__ == "__main__":
    run_ingestion()

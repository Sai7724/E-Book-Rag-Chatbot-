# Asks the running API 6 questions and prints what comes back.
# Start the server first (uvicorn app:app), then run:  python tests_sample_queries.py
# The eBook doesn't cover the last two questions, so the bot should refuse them.
import json
import time
import urllib.request

from src.config import REFUSAL

API_URL = "http://127.0.0.1:8000/chat"

IN_BOOK = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
]
OUT_OF_BOOK = [
    "Who won the 2022 FIFA World Cup?",
    "What is the current price of Bitcoin?",
]


def ask(question):
    # POST the question to /chat and return the JSON reply.
    body = json.dumps({"query": question}).encode()
    req = urllib.request.Request(API_URL, body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


for q in IN_BOOK + OUT_OF_BOOK:
    # Print the answer, its score and the start of the top chunk.
    # The 5-second pause keeps us under Gemini's free per-minute limit.
    time.sleep(5)
    r = ask(q)
    print(f"\nQ: {q}\nA: {r['answer']}\nScore: {r['confidence_score']}")
    print(f"Top chunk: {r['retrieved_chunks'][0][:150]}...")

    # Off-topic questions have to get the refusal and a score of 0.
    if q in OUT_OF_BOOK:
        assert REFUSAL in r["answer"] and r["confidence_score"] == 0.0, f"Not grounded: {q}"

print("\nAll 6 queries ran, and both off-topic questions were refused.")

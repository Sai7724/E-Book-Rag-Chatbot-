from typing import List, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph

from src.config import EMBED_DIM, EMBED_MODEL, LLM_MODELS, PINECONE_INDEX_NAME, REFUSAL, TOP_K


# State passed between the nodes. Each node returns the fields it fills in.
class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


# init clients
embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, output_dimensionality=EMBED_DIM)
vector_store = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
models = [ChatGoogleGenerativeAI(model=m, max_retries=1) for m in LLM_MODELS]
llm = models[0].with_fallbacks(models[1:])

PROMPT = """You are a strict assistant. Answer the question using ONLY the context below.
Do not use any outside knowledge. If the context is relevant, answer from it even if
it only partly covers the question. If nothing in the context is relevant,
reply exactly: "{refusal}"

Context:
{context}

Question: {question}"""


def retrieve(state: AgentState):
    results = vector_store.similarity_search_with_score(state["question"], k=TOP_K)
    context = [doc.page_content for doc, _ in results]
    
    # calculate avg score
    total_score = sum(s for _, s in results)
    score = total_score / len(results) if results else 0.0
    
    return {"context": context, "score": round(score, 3)}


def generate(state: AgentState):
    ctx_str = "\n\n---\n\n".join(state["context"])
    prompt = PROMPT.format(refusal=REFUSAL, context=ctx_str, question=state["question"])
    
    ans = llm.invoke(prompt).text
    
    # drop score if model refused
    final_score = 0.0 if REFUSAL in ans else state["score"]
    return {"answer": ans, "score": final_score}


workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
graph = workflow.compile()

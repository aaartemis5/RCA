# fastapi_rag.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

# RAG imports (unchanged)
from retriever import get_retriever
from prompt_llm import build_prompt as build_rag_prompt, get_llm_response as rag_llm_response
from prompt_llm_description import build_summary_prompt, get_llm_response as summary_llm_response
from config import EMBEDDING_MODEL, OPENAI_CHAT_MODEL, OPENAI_API_KEY

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Initialize retriever
try:
    retriever = get_retriever(k_results=9)
    logging.info("Retriever initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize retriever: {e}")
    raise RuntimeError("Retriever init failed")

# Initialize FastAPI
app = FastAPI()

# ----------------- RAG Endpoint -----------------
class QueryInput(BaseModel):
    query: str

@app.post("/rag-query/")
def handle_query(input_data: QueryInput) -> Dict[str, Any]:
    user_query = input_data.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Empty query received")
    logging.info(f"Received RAG query: {user_query}")

    # Retrieve documents
    try:
        retrieved_docs = retriever.invoke(user_query)
    except Exception as e:
        logging.error(f"Retriever error: {e}")
        raise HTTPException(status_code=500, detail="Retriever failed")

    # Format context chunks
    context_chunks: List[Dict[str, Any]] = []
    for i, doc in enumerate(retrieved_docs):
        text = getattr(doc, "page_content", None)
        metadata = getattr(doc, "metadata", {}) or {}
        if text:
            context_chunks.append({
                "id": metadata.get("id", f"chunk_{i}"),
                "text_to_embed": text,
                "metadata": metadata,
            })

    # Build RAG prompt
    try:
        final_prompt = build_rag_prompt(user_query, context_chunks)
    except Exception as e:
        logging.error(f"Prompt build error: {e}")
        raise HTTPException(status_code=500, detail="Prompt construction failed")

    # Get LLM answer for RAG
    try:
        answer = rag_llm_response(final_prompt)
    except Exception as e:
        logging.error(f"LLM error: {e}")
        raise HTTPException(status_code=500, detail="LLM processing failed")

    return {"query": user_query, "answer": answer}

# ------------- Insight Summary Endpoint -------------
class InsightData(BaseModel):
    name: str
    description: str = ""
    assetName: str
    insightType: str
    priority: str
    insightStatus: str
    timeDiff: str
    timesInsightWasOpened: int
    equipments: List[str]
    richTextContent: str

class InsightPayload(BaseModel):
    message: str
    data: InsightData
    Comments: Any
    Insights: Any

@app.post("/generate-description")
async def generate_description(payload: InsightPayload) -> Dict[str, str]:
    try:
        # Convert to dict and build prompt
        insight_dict = payload.data.dict()
        prompt = build_summary_prompt(insight_dict)

        # Get summary
        summary = summary_llm_response(prompt)

        # Return only the description JSON
        return {"description": summary}

    except Exception as e:
        logging.error(f"Failed to generate description: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ Run Uvicorn ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_rag:app", host="0.0.0.0", port=8000, reload=True)

# fastapi_rag.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import json

from retriever import get_retriever
from prompt_llm import build_prompt, get_llm_response
from config import EMBEDDING_MODEL

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Initialize retriever
try:
    retriever = get_retriever(k_results=9)
    logging.info("Retriever initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize retriever: {e}")
    raise RuntimeError("Retriever init failed")

# FastAPI app
app = FastAPI()

# Define request model
class QueryInput(BaseModel):
    query: str

# POST endpoint
@app.post("/rag-query/")
def handle_query(input_data: QueryInput):
    user_query = input_data.query.strip()

    if not user_query:
        raise HTTPException(status_code=400, detail="Empty query received")

    logging.info(f"Received query: {user_query}")

    # Step 1: Retrieve documents
    try:
        retrieved_docs: List[Any] = retriever.invoke(user_query)
    except Exception as e:
        logging.error(f"Retriever error: {e}")
        raise HTTPException(status_code=500, detail="Retriever failed")

    # Step 2: Format docs
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

    # Step 3: Build prompt
    try:
        final_prompt = build_prompt(user_query, context_chunks)
    except Exception as e:
        logging.error(f"Prompt build error: {e}")
        raise HTTPException(status_code=500, detail="Prompt construction failed")

    # Step 4: Get LLM answer
    try:
        answer = get_llm_response(final_prompt)
    except Exception as e:
        logging.error(f"LLM error: {e}")
        raise HTTPException(status_code=500, detail="LLM processing failed")

    # Step 5: Return answer
    return {"query": user_query, "answer": answer}

 
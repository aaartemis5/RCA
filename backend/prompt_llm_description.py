import logging
import openai
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL_descp

# ------------------ Logging Setup ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------ Initialize OpenAI Client ------------------
openai_client = None
try:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in config.py or environment variables.")
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    logging.info("OpenAI client initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing OpenAI client: {e}")

# ------------------ RAG Prompt Builder ------------------
def build_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Builds RAG prompt for RCA based on provided context chunks.
    (Unchanged STP RCA prompt builder)
    """
    formatted_string = ""
    for i, chunk in enumerate(context_chunks):
        text = chunk.get("text_to_embed", "N/A")
        meta = chunk.get("metadata", {})
        heading = meta.get("heading", "N/A")
        page = meta.get("page_number", meta.get("start_page", "N/A"))
        source = meta.get("source_document", "N/A")
        formatted_string += f"--- Context Chunk {i+1} ---\n"
        formatted_string += f"Source: {source}\nPage: {page}\nHeading: {heading}\nText: {text}\n---------------------------\n\n"
    prompt_template = """
You are an expert AI assistant specializing in Root Cause Analysis (RCA) for Sewage Treatment Plant (STP) operations and troubleshooting. Your knowledge comes *exclusively* from the provided context snippets.

**Goal:** Identify potential root cause(s) of the problem described in the query using only the context.

**Instructions:**
1. Summary of Likely Root Cause(s): List 2–4 possible causes.
2. Detailed Root Cause Analysis with sections: Problem Summary, Potential Immediate Causes, Potential Root Causes & Reasoning, Recommended Diagnostic Steps.
3. References: List each chunk's source, heading, and page.
4. State 'Insufficient Context' if needed, then optional General Knowledge.

Context Snippets:
---------------------
{context}
---------------------

User Query: {query}

**Root Cause Analysis:**"""
    template = PromptTemplate(input_variables=["query", "context"], template=prompt_template)
    return template.format(query=query, context=formatted_string)

# ------------------ Summary Prompt Builder ------------------
def build_summary_prompt(insight_data: Dict[str, Any]) -> str:
    """
    Constructs a prompt for generating a 2–3 line summary of an insight.
    """
    prompt = f"""
You are an AI assistant that writes concise, 2–3 sentence summaries of maintenance or incident insights.
Based on the structured data below, generate a clear description highlighting what happened, where, and why it matters.

- Title: {insight_data.get('name', '')}
- Asset: {insight_data.get('assetName', '')}
- Type: {insight_data.get('insightType', '')}
- Priority: {insight_data.get('priority', '')}
- Status: {insight_data.get('insightStatus', '')}
- Reported At: {insight_data.get('timeDiff', '')}
- Occurrences: {insight_data.get('timesInsightWasOpened', '')}
- Equipment: {', '.join(insight_data.get('equipments', []))}
- Details: {insight_data.get('richTextContent', '')}

Summary:""".strip()
    return prompt

# ------------------ LLM Caller ------------------
def get_llm_response(prompt: str) -> str:
    """Sends the prompt to OpenAI and returns the generated text."""
    if not openai_client:
        raise RuntimeError("OpenAI client not initialized.")
    try:
        logging.info(f"Sending prompt to OpenAI model: {OPENAI_CHAT_MODEL_descp}")
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL_descp,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        raise RuntimeError(f"LLM error: {str(e)}")

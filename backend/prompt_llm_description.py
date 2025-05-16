import logging
import openai
from typing import Dict, Any
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
    """Sends the summary prompt to OpenAI and returns the generated description."""
    if not openai_client:
        raise RuntimeError("OpenAI client not initialized.")
    try:
        logging.info(f"Sending summary prompt to OpenAI model: {OPENAI_CHAT_MODEL_descp}")
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL_descp,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates concise incident summaries."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Summary LLM call failed: {e}")
        raise RuntimeError(f"LLM error: {str(e)}")

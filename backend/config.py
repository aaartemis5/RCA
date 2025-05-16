# config.py
import os
from dotenv import load_dotenv
import logging # Optional: Add logging for config loading

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Loading configuration from environment variables...")

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rca-openai") # Provide default
logging.info(f"Pinecone Index Name: {PINECONE_INDEX_NAME}")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # Load if needed

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Specify the embedding model to be used by index.py and retriever.py
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
# Specify the chat model to be used by prompt_llm.py (if using OpenAI)
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
OPENAI_CHAT_MODEL_descp = os.getenv("OPENAI_CHAT_MODEL_descp")
logging.info(f"OpenAI Embedding Model: {OPENAI_EMBEDDING_MODEL}")
logging.info(f"OpenAI Chat Model: {OPENAI_CHAT_MODEL}")

# --- Groq Configuration (Optional - Load only if needed) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
if GROQ_API_KEY:
    logging.info(f"Groq Model Name: {GROQ_MODEL_NAME}")
else:
    logging.info("Groq API Key not found, Groq LLM will not be available.")

# --- MongoDB Configuration ---
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "rca_db") # Default DB name
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "chats") # Default Collection name
logging.info(f"MongoDB DB Name: {MONGODB_DB_NAME}")
logging.info(f"MongoDB Collection Name: {MONGODB_COLLECTION_NAME}")

# --- Hugging Face Configuration (Optional) ---
# HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# if HUGGINGFACE_API_KEY:
#     logging.info("Hugging Face API Key loaded.")


# --- Validation (Simplified) ---
essential_keys = {
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "OPENAI_EMBEDDING_MODEL": OPENAI_EMBEDDING_MODEL,
    "OPENAI_CHAT_MODEL": OPENAI_CHAT_MODEL, # Assuming OpenAI is used for chat too
    "MONGODB_URI": MONGODB_URI,
    "MONGODB_DB_NAME": MONGODB_DB_NAME,
    "MONGODB_COLLECTION_NAME": MONGODB_COLLECTION_NAME,
}

missing_keys = [key for key, value in essential_keys.items() if not value]

if missing_keys:
    logging.error(f"CRITICAL CONFIGURATION ERROR: Missing essential environment variables: {', '.join(missing_keys)}")
    # Depending on your deployment, you might want to raise an exception or exit here
    # raise ValueError(f"Missing essential config: {', '.join(missing_keys)}")
else:
    logging.info("Essential configuration variables loaded.")

# --- EXPORT Aliases (for compatibility with old variable names if needed temporarily) ---
# These allow existing code using the old names to potentially still work,
# but it's better to update the code to use the new names directly.

EMBEDDING_MODEL = OPENAI_EMBEDDING_MODEL # Make EMBEDDING_MODEL point to the OpenAI one

# GROQ_MODEL = GROQ_MODEL_NAME # If using Groq
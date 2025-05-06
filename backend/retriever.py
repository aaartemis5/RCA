# backend/retriever.py
import os
# Use specific OpenAI/Pinecone config variables
from config import PINECONE_INDEX_NAME, PINECONE_API_KEY, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
from langchain_community.vectorstores import Pinecone as LangchainPinecone
# Use Langchain's OpenAI wrapper for embeddings
from langchain_openai import OpenAIEmbeddings
# from pinecone import Pinecone as BasePinecone # Only needed for direct debug query function if used
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Optional Debug Query Function (Keep commented out unless needed)
# def debug_pinecone_query(...): ...

def get_retriever(k_results=5):
    """
    Initializes and returns a Langchain retriever using OpenAI Embeddings
    for the configured Pinecone index.
    """
    logging.info(f"[retriever.py] Initializing retriever for index '{PINECONE_INDEX_NAME}'...")

    # --- Initialize OpenAI Embeddings via Langchain ---
    embeddings = None
    try:
        logging.info(f"[retriever.py] Loading OpenAI embedding model via Langchain: {OPENAI_EMBEDDING_MODEL}")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in config/environment.")
        # Initialize the Langchain wrapper for OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
            # Add other parameters like chunk_size if needed by the specific model/API version
        )
        logging.info("[retriever.py] OpenAI Embedding model loaded via Langchain.")
    except Exception as e:
        logging.error(f"[retriever.py] Failed to load OpenAI embedding model via Langchain: {e}")
        raise # Critical failure if embeddings don't load

    # --- Initialize Langchain Pinecone Vector Store ---
    try:
        logging.info(f"[retriever.py] Connecting to Langchain Pinecone vector store for index: {PINECONE_INDEX_NAME}")
        # This key MUST match the key used in index.py to store the text in metadata
        text_key_in_metadata = 'chunk_text'
        logging.info(f"[retriever.py] Configuring Langchain to use metadata key '{text_key_in_metadata}' for text content.")

        vector_store = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME, # Use variable from config
            embedding=embeddings, # Use the OpenAI embeddings object
            text_key=text_key_in_metadata
        )
        logging.info("[retriever.py] Langchain Pinecone vector store connected.")
    except Exception as e:
        logging.error(f"[retriever.py] Failed to initialize Langchain Pinecone vector store: {e}")
        raise # Critical failure

    # --- Create Retriever ---
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_results}
        )
        logging.info(f"[retriever.py] Retriever created with k={k_results}.")
        return retriever
    except Exception as e:
        logging.error(f"[retriever.py] Failed to create retriever from vector store: {e}")
        raise # Critical failure


# --- Test Block ---
if __name__ == "__main__":
    print("\n--- Testing retriever.py Standalone with OpenAI Embeddings ---")
    try:
        # Get retriever instance (debug query is off by default)
        retriever_instance = get_retriever(k_results=3)

        # Use a query relevant to your indexed data
        test_query = "What are potential causes for valve failure?" # Example query

        print(f"\n[Test Block] Running test query: '{test_query}'")
        # Invoke the retriever instance
        results = retriever_instance.invoke(test_query)
        print(f"[Test Block] Langchain Retriever returned {len(results)} results.")

        print("\n[Test Block] Retrieved Chunks & Content:")
        if not results:
            print("  No results found by Langchain retriever.")
        for i, doc in enumerate(results):
            print(f"\n  Chunk {i+1} (Langchain Document):")
            metadata = getattr(doc, 'metadata', {})
            page_content = getattr(doc, 'page_content', None)
            print(f"    Metadata: {str(metadata)[:500]}...") # Print metadata snippet
            if page_content:
                print(f"    page_content (Text Used by Langchain):\n{'-'*10}\n{page_content}\n{'-'*10}")
            else:
                # This indicates Langchain failed to extract text using the specified text_key
                print(f"    WARNING: page_content attribute missing or empty. Check if 'chunk_text' key exists and has text in Pinecone metadata.")
            print("-" * 10)

    except Exception as e:
        print(f"\n[Test Block] An error occurred during testing: {e}")
        logging.exception("[Test Block] Error details:")
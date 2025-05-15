# terminal_rag.py

import logging
import os
import json # For potentially viewing metadata if needed

# Import necessary components from your existing modules
from retriever import get_retriever # Assumes get_retriever is correctly configured
from prompt_llm import build_prompt, get_llm_response
from config import EMBEDDING_MODEL # Required for logging, retriever init implicitly uses it
from typing import List, Dict, Any # For type hinting

# --- Logging Setup ---
# Keep INFO level to see retriever/LLM logs, but terminal output is primary
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Initialize Core Components ---
retriever = None
try:
    # Initialize retriever (adjust k_results if desired)
    retriever = get_retriever(k_results=9) # Fetch top 9 chunks
    logging.info("Retriever initialized successfully.")
except Exception as e:
    logging.error(f"CRITICAL: Failed to initialize retriever: {e}", exc_info=True)
    print("\nERROR: Could not initialize the document retriever. Exiting.")
    exit(1)
# Note: Embedding model is loaded within get_retriever
# Note: LLM client is initialized within prompt_llm.py when get_llm_response is called

def run_terminal_chat():
    """Runs the interactive terminal-based RAG loop."""
    print("\n--- STP Root Cause Analysis Assistant (Terminal Mode) ---")
    print("Enter your query about an STP problem, or type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_query = input("\nYou: ")
            if user_query.lower() in ['quit', 'exit']:
                print("Exiting assistant. Goodbye!")
                break
            if not user_query.strip():
                print("Please enter a query.")
                continue

            logging.info(f"Processing query: '{user_query}'")

            # 1. Retrieve Context using Langchain Retriever
            print("Assistant: Retrieving relevant information...")
            try:
                # This returns Langchain Document objects
                retrieved_docs: List[Any] = retriever.invoke(user_query)
                logging.info(f"Retriever invoked. Returned {len(retrieved_docs)} Langchain Document objects.")

                if not retrieved_docs:
                    print("Assistant: Found no specific documents matching the query in the knowledge base.")
            except Exception as e:
                logging.error(f"Error during retrieval: {e}", exc_info=True)
                print("Assistant: Sorry, I encountered an error while retrieving information.")
                continue

            # 2. Format Context for Prompting (using the retrieved docs)
            context_chunks_for_prompt: List[Dict[str, Any]] = []
            if retrieved_docs:
                logging.info(f"Formatting {len(retrieved_docs)} retrieved docs for LLM prompt...")
                # <<< --- ADDED PRINT STATEMENTS FOR FULL RETRIEVED CHUNK DETAILS --- >>>
                print("\n" + "="*15 + " START: FULL RETRIEVED CONTEXT CHUNKS " + "="*15)
                for i, doc in enumerate(retrieved_docs):
                    try:
                        text_content = getattr(doc, 'page_content', None)
                        metadata = getattr(doc, 'metadata', {})
                        if not isinstance(metadata, dict): metadata = {}
                        chunk_id = metadata.get('id', f"retrieved_doc_{i}")

                        print(f"\n--- Retrieved Chunk {i+1} (ID: {chunk_id}) ---")
                        print(f"  Metadata: {json.dumps(metadata, indent=2)}") # Pretty print metadata
                        if text_content and isinstance(text_content, str):
                            print(f"  Full Text (page_content):\n{text_content}")
                            context_chunks_for_prompt.append({
                                "id": chunk_id,
                                "text_to_embed": text_content,
                                "metadata": metadata
                            })
                        else:
                            print(f"  WARNING: Langchain Document {i} (ID: {chunk_id}) has missing/invalid page_content.")
                            logging.warning(f"Langchain Document {i} (ID: {chunk_id}) has missing/invalid page_content. This chunk WILL NOT be sent to LLM.")
                        print(f"--- End of Chunk {i+1} ---")
                    except Exception as e:
                        logging.error(f"Error processing retrieved document {i} before formatting: {e}")
                print("="*15 + " END: FULL RETRIEVED CONTEXT CHUNKS " + "="*15 + "\n")
                # <<< --- END OF ADDED PRINT STATEMENTS --- >>>

            if not context_chunks_for_prompt:
                 print("Assistant: Although matches might have been found, no valid text content could be extracted to analyze. Please check retriever configuration ('text_key') and indexed data.")
                 # pass # Let it proceed, build_prompt handles empty list

            print(f"Assistant: Using {len(context_chunks_for_prompt)} valid context snippets for analysis...")

            # 3. Build Prompt
            try:
                final_prompt = build_prompt(user_query, context_chunks_for_prompt)
                logging.info(f"Generated prompt for LLM (using {len(context_chunks_for_prompt)} chunks).")

                # <<< --- PRINT THE ENTIRE FINAL PROMPT SENT TO LLM --- >>>
                print("\n" + "="*20 + " FULL PROMPT BEING SENT TO LLM " + "="*20)
                print(final_prompt)
                print("="*20 + " END OF FULL PROMPT " + "="*20 + "\n")
                # <<< --------------------------------------------------- >>>

            except Exception as e:
                logging.error(f"Error building prompt: {e}", exc_info=True)
                print("Assistant: Sorry, I encountered an error preparing the analysis request.")
                continue

            # 4. Get LLM Response
            print("Assistant: Analyzing... (Sending request to LLM)")
            try:
                answer = get_llm_response(final_prompt)
            except Exception as e:
                logging.error(f"Error getting LLM response: {e}", exc_info=True)
                print("Assistant: Sorry, I encountered an error during the analysis.")
                continue

            # 5. Print Answer
            print("\nAssistant (LLM Response):") # Clarified it's the LLM's response
            print(answer)
            print("-" * 30) # Separator for next query

        except KeyboardInterrupt:
            print("\nExiting assistant. Goodbye!")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            print("Assistant: An unexpected error occurred. Please try again.")

# --- Main Execution Guard ---
if __name__ == "__main__":
    if retriever is None:
        logging.critical("Retriever is not initialized. Cannot start chat.")
    else:
        run_terminal_chat()
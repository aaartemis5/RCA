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
    retriever = get_retriever(k_results=5) # Fetch top 5 chunks
    logging.info("Retriever initialized successfully.")
except Exception as e:
    logging.error(f"CRITICAL: Failed to initialize retriever: {e}", exc_info=True)
    print("\nERROR: Could not initialize the document retriever. Exiting.")
    exit(1)
# Note: Embedding model is loaded within get_retriever
# Note: Groq client is initialized within prompt_llm.py when get_llm_response is called

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

                # <<<--- ADDED DEBUG LOGGING for Raw Retrieved Docs --- >>>
                if retrieved_docs:
                    print("\n--- DEBUG: Raw Retrieved Metadata from Langchain Docs ---")
                    for i, doc in enumerate(retrieved_docs):
                        raw_metadata = getattr(doc, 'metadata', 'METADATA ATTRIBUTE MISSING')
                        page_content_exists = hasattr(doc, 'page_content') and doc.page_content is not None and doc.page_content != ""
                        print(f"  Doc {i+1} Metadata: {str(raw_metadata)[:500]}...") # Print metadata snippet
                        # Explicitly check for the key retriever.py uses ('chunk_text' assumed)
                        if isinstance(raw_metadata, dict):
                            print(f"    Metadata contains 'chunk_text'? {'chunk_text' in raw_metadata}")
                            # print(f"    Metadata contains 'text_snippet'? {'text_snippet' in raw_metadata}") # Uncomment if relevant
                        else:
                             print(f"    Metadata type is not dict: {type(raw_metadata)}")
                        print(f"    Doc has valid page_content attribute? {page_content_exists}")
                        if page_content_exists:
                             print(f"    page_content snippet: {doc.page_content[:100]}...")
                    print("--- END DEBUG ---")
                # <<<--------------------------------------------------->>>

                if not retrieved_docs:
                    print("Assistant: Found no specific documents matching the query in the knowledge base.")
                    # Decide if you want to proceed without context
                    # continue # Option 1: Ask for another query
                    # pass # Option 2: Let it proceed to LLM with empty context list

            except Exception as e:
                logging.error(f"Error during retrieval: {e}", exc_info=True)
                print("Assistant: Sorry, I encountered an error while retrieving information.")
                continue

            # 2. Format Context for Prompting (using the retrieved docs)
            context_chunks_for_prompt: List[Dict[str, Any]] = []
            if retrieved_docs:
                logging.info(f"Formatting {len(retrieved_docs)} retrieved docs for LLM prompt...")
                for i, doc in enumerate(retrieved_docs):
                    try:
                        # Text content SHOULD be in page_content if text_key was correct
                        text_content = getattr(doc, 'page_content', None)
                        metadata = getattr(doc, 'metadata', {})
                        if not isinstance(metadata, dict): metadata = {}

                        # Use the 'id' stored in metadata during indexing, or create fallback
                        chunk_id = metadata.get('id', f"retrieved_doc_{i}") # Use ID from metadata

                        if text_content and isinstance(text_content, str):
                            # Add to list for build_prompt
                            context_chunks_for_prompt.append({
                                "id": chunk_id,
                                "text_to_embed": text_content, # Use the text Langchain extracted
                                "metadata": metadata # Pass original metadata along
                            })
                        else:
                            # This directly relates to the warning you saw earlier
                            logging.warning(f"Langchain Document {i} (ID: {chunk_id}) has missing/invalid page_content. This chunk WILL NOT be sent to LLM.")
                    except Exception as e:
                        logging.error(f"Error processing retrieved document {i} before formatting: {e}")

            # Check if any chunks were successfully formatted
            if not context_chunks_for_prompt:
                 print("Assistant: Although matches were found, could not extract valid text content to analyze. Please check retriever configuration ('text_key') and indexed data.")
                 # Optionally, you could still send the query to the LLM without context.
                 # Let's continue for now, build_prompt will handle empty list.
                 pass


            print(f"Assistant: Found {len(context_chunks_for_prompt)} valid context snippets. Preparing analysis...")


            # 3. Build Prompt
            try:
                final_prompt = build_prompt(user_query, context_chunks_for_prompt)
                logging.info(f"Generated prompt for LLM (using {len(context_chunks_for_prompt)} chunks).")
            except Exception as e:
                logging.error(f"Error building prompt: {e}", exc_info=True)
                print("Assistant: Sorry, I encountered an error preparing the analysis request.")
                continue

            # 4. Get LLM Response
            print("Assistant: Analyzing...")
            try:
                answer = get_llm_response(final_prompt)
            except Exception as e:
                logging.error(f"Error getting LLM response: {e}", exc_info=True)
                print("Assistant: Sorry, I encountered an error during the analysis.")
                continue

            # 5. Print Answer
            print("\nAssistant:")
            print(answer)
            print("-" * 30) # Separator for next query

        except KeyboardInterrupt:
            print("\nExiting assistant. Goodbye!")
            break
        except Exception as e:
            # Catch any other unexpected errors in the loop
            logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            print("Assistant: An unexpected error occurred. Please try again.")

# --- Main Execution Guard ---
if __name__ == "__main__":
    # Ensure retriever is available before starting chat
    if retriever is None:
        logging.critical("Retriever is not initialized. Cannot start chat.")
    else:
        run_terminal_chat()
# backend/main.py

from retriever import get_retriever # Assumes this retrieves Langchain Document objects
from prompt_llm import build_prompt, get_llm_response, format_context # Import format_context if you want to log it
import json
import os
import logging
from typing import List, Dict, Any # For type hinting

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # --- Initialize Retriever ---
    try:
        # Get retriever. k_results determines how many chunks are fetched. Adjust as needed.
        # More chunks = more context, but longer prompts & potentially more noise.
        retriever = get_retriever(k_results=5) # Fetch top 5 chunks
        logging.info("[main.py] Retriever initialization attempted.")
    except Exception as e:
        logging.error(f"[main.py] Failed to initialize retriever: {e}", exc_info=True)
        print("CRITICAL: Could not initialize the document retriever. Exiting.")
        return

    # --- Get User Query ---
    # Updated prompt for the user
    user_query = input("Describe the Sewage Treatment Plant problem: ")
    if not user_query:
        print("Problem description cannot be empty.")
        return
    logging.info(f"[main.py] Processing query: '{user_query}'")

    # 1. Retrieve relevant CHUNKS from Pinecone/Vectorstore
    try:
        # retriever.invoke should return a list of Langchain Document objects
        retrieved_docs: List[Any] = retriever.invoke(user_query) # Type hint for clarity
        logging.info(f"[main.py] Retriever invoked. Found {len(retrieved_docs)} relevant document chunks.")
    except Exception as e:
        logging.error(f"[main.py] Error during retriever invocation: {e}", exc_info=True)
        print("Error retrieving relevant information from the knowledge base.")
        return

    # 2. Prepare context list for the prompt builder
    context_chunks_for_prompt: List[Dict[str, Any]] = [] # This will hold the dicts for build_prompt
    print("\n--- Processing Retrieved Information ---")
    if not retrieved_docs:
        print("No relevant information found in the knowledge base for this problem.")
        # Optionally, still call the LLM with no context or exit
        # For now, let's continue but context will be empty

    for i, doc in enumerate(retrieved_docs):
        print(f"\nProcessing retrieved chunk {i+1}:")
        try:
            # Extract text (page_content)
            text_content = getattr(doc, 'page_content', None)
            # Extract metadata dictionary
            metadata = getattr(doc, 'metadata', {}) # Default to empty dict

            if not isinstance(metadata, dict):
                 logging.warning(f"Metadata for retrieved chunk {i+1} is not a dictionary. Type: {type(metadata)}. Skipping metadata details.")
                 metadata = {} # Ensure metadata is a dict

            # Extract specific metadata fields (adjust keys based on what's actually stored)
            # Assuming the ID used during indexing is stored in metadata['id']
            # If not, you might need to generate one or rely on vector ID if accessible
            chunk_id = metadata.get('id', f"retrieved_doc_{i+1}") # Fallback ID
            source_doc = metadata.get('source_document', 'Unknown Source')
            page = metadata.get('page_number', 'N/A')
            heading = metadata.get('heading', 'N/A')

            print(f"  Source: {source_doc}, Page: {page}, Heading: {heading}")

            if text_content and isinstance(text_content, str):
                print(f"  Text Snippet: {text_content[:250]}...")
                # Create the dictionary in the format expected by build_prompt
                chunk_dict = {
                    "id": chunk_id,
                    "text_to_embed": text_content, # Key expected by prompt_llm's format_context
                    "metadata": metadata          # Pass the whole original metadata dict
                }
                context_chunks_for_prompt.append(chunk_dict)
            else:
                print(f"  WARNING: Missing or invalid text content for retrieved chunk {i+1}.")

        except Exception as e:
            logging.error(f"Error processing retrieved document chunk {i+1}: {e}", exc_info=True)
            print(f"An error occurred while processing retrieved chunk {i+1}.")

    # Check if any usable context was gathered
    if not context_chunks_for_prompt:
        logging.warning("[main.py] No usable context could be extracted from retrieved chunks.")
        print("\nWARNING: Could not extract usable context to send to the AI.")
        # Decide how to proceed - maybe inform the user and exit?
        # Or send the query to the LLM without specific context?
        # Let's inform the user and stop for now.
        print("Cannot proceed without context.")
        return

    # Optional: Log the formatted context that will be sent (can be verbose)
    # formatted_log_context = format_context(context_chunks_for_prompt)
    # logging.info(f"[main.py] Formatted context being sent to LLM:\n{formatted_log_context[:500]}...")

    # 3. Build the final prompt for the LLM using the list of context dicts
    try:
        final_prompt = build_prompt(user_query, context_chunks_for_prompt)
        print("\n--- Preparing Request for AI Analysis ---")
        # print(final_prompt[:800] + "...") # Avoid printing too much potentially sensitive context
        logging.info("[main.py] Final prompt built successfully.")
        print("-" * 20)
    except Exception as e:
        logging.error(f"[main.py] Error building prompt: {e}", exc_info=True)
        print("Error occurred while preparing the request for the AI analysis.")
        return

    # 4. Get LLM response
    print("\n--- Requesting LLM Response for RCA ---")
    try:
        answer = get_llm_response(final_prompt)
        print("\n--- AI Root Cause Analysis ---")
        print(answer)
    except Exception as e:
        logging.error(f"[main.py] Error getting LLM response: {e}", exc_info=True)
        print("Error occurred while getting the analysis from the AI.")

if __name__ == "__main__":
    main()
# batch_rca_from_csv.py
import pandas as pd  
import logging
import time
import os
from tqdm import tqdm
import json # Needed for tracking file

# ... (imports for retriever, prompt_llm, config) ...
from retriever import get_retriever
from prompt_llm import build_prompt, get_llm_response
from typing import List, Dict, Any

# --- Configuration ---
INPUT_CSV_PATH = "backend/finalcsv.csv"
OUTPUT_CSV_PATH = "output_rca_results3.csv"
QUERY_COLUMN_NAME = "heading.eng"
ANSWER_COLUMN_NAME = "RCA_Analysis"
# Set NUM_ROWS_TO_PROCESS to 0 or a large number if you want it to process all *remaining* rows
NUM_ROWS_TO_PROCESS = 100 # Process UP TO this many *new* rows per run
DELAY_BETWEEN_QUERIES = 1.5 # Adjusted delay slightly
# Tracking file for processed input CSV row indices
PROCESSED_ROWS_TRACKING_FILE = "backend/processed_input_rows.json"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Core Components ---
# ... (Retriever initialization - keep as before) ...
retriever = None
try:
    retriever = get_retriever(k_results=5)
    logging.info("Retriever initialized successfully.")
except Exception as e:
    logging.error(f"CRITICAL: Failed to initialize retriever: {e}", exc_info=True); exit(1)

# --- Load Processed Row Indices ---
processed_row_indices = set()
if os.path.exists(PROCESSED_ROWS_TRACKING_FILE):
    try:
        with open(PROCESSED_ROWS_TRACKING_FILE, "r", encoding='utf-8') as f:
            indices_list = json.load(f)
            if isinstance(indices_list, list):
                processed_row_indices = set(indices_list) # Load indices
            else:
                logging.warning(f"Format error in {PROCESSED_ROWS_TRACKING_FILE}. Starting fresh.")
        logging.info(f"Loaded {len(processed_row_indices)} already processed row indices from {PROCESSED_ROWS_TRACKING_FILE}.")
    except Exception as e:
        logging.error(f"Error loading {PROCESSED_ROWS_TRACKING_FILE}: {e}. Starting fresh.")
else:
    logging.info(f"No processed row tracking file ({PROCESSED_ROWS_TRACKING_FILE}) found. Starting fresh.")


# --- get_rca_for_query function (keep as before) ---
def get_rca_for_query(query: str) -> str:
    # ... (implementation remains the same) ...
    if not query or not isinstance(query, str) or not query.strip(): return "ERROR: Invalid query provided."
    context_chunks_for_prompt: List[Dict[str, Any]] = []
    try: # Retrieval
        # logging.info(f"Retrieving context for query: '{query[:100]}...'") # Reduce log noise
        retrieved_docs = retriever.invoke(query); # logging.info(f"Retrieved {len(retrieved_docs)} chunks.")
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                text_content = getattr(doc, 'page_content', None); metadata = getattr(doc, 'metadata', {}); chunk_id = metadata.get('id', f"retrieved_doc_{i}")
                if not isinstance(metadata, dict): metadata = {}
                if text_content and isinstance(text_content, str): context_chunks_for_prompt.append({"id": chunk_id, "text_to_embed": text_content, "metadata": metadata})
                # else: logging.warning(f"Retrieved chunk {i} (ID: {chunk_id}) has no valid text content. Skipping.") # Reduce log noise
        # if not context_chunks_for_prompt: logging.warning("No usable context chunks found after retrieval.") # Reduce log noise
    except Exception as e: logging.error(f"Error during retrieval for query '{query[:50]}...': {e}"); return f"ERROR: Failed during context retrieval - {e}" # Keep error logs
    try: # Prompt Building
        final_prompt = build_prompt(query, context_chunks_for_prompt); # logging.info("Prompt built successfully.") # Reduce log noise
    except Exception as e: logging.error(f"Error building prompt for query '{query[:50]}...': {e}"); return f"ERROR: Failed during prompt building - {e}" # Keep error logs
    try: # LLM Call
        # logging.info("Requesting LLM response...") # Reduce log noise
        answer = get_llm_response(final_prompt); # logging.info("LLM response received.") # Reduce log noise
        return answer
    except Exception as e: logging.error(f"Error getting LLM response for query '{query[:50]}...': {e}"); return f"ERROR: Failed during LLM call - {e}" # Keep error logs


def process_csv():
    """Loads CSV, processes rows NOT already processed, appends results."""
    # --- Load Input CSV ---
    if not os.path.exists(INPUT_CSV_PATH): logging.error(f"Input CSV not found: {INPUT_CSV_PATH}"); return
    try: df_input = pd.read_csv(INPUT_CSV_PATH); logging.info(f"Loaded {len(df_input)} rows from {INPUT_CSV_PATH}")
    except Exception as e: logging.error(f"Error loading input CSV: {e}"); return
    if QUERY_COLUMN_NAME not in df_input.columns: logging.error(f"Query column '{QUERY_COLUMN_NAME}' not found. Available: {list(df_input.columns)}"); return

    # --- Read Existing Output (or create empty DataFrame) ---
    if os.path.exists(OUTPUT_CSV_PATH):
        try:
            df_output_existing = pd.read_csv(OUTPUT_CSV_PATH)
            logging.info(f"Loaded {len(df_output_existing)} existing results from {OUTPUT_CSV_PATH}")
        except Exception as e:
            logging.warning(f"Could not read existing output file {OUTPUT_CSV_PATH}: {e}. Starting fresh output.")
            df_output_existing = pd.DataFrame(columns=[QUERY_COLUMN_NAME, ANSWER_COLUMN_NAME])
    else:
        logging.info(f"Output file {OUTPUT_CSV_PATH} not found. Creating new one.")
        df_output_existing = pd.DataFrame(columns=[QUERY_COLUMN_NAME, ANSWER_COLUMN_NAME])

    # --- Process Rows ---
    new_results = [] # Store results from THIS run
    rows_processed_this_run = 0
    rows_skipped_already_done = 0
    rows_skipped_invalid_query = 0

    logging.info(f"Scanning input CSV to process up to {NUM_ROWS_TO_PROCESS} new rows...")

    # Iterate through the entire input dataframe using index
    for index, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Scanning Input Rows"):

        # Check if we have reached the limit for this run
        if NUM_ROWS_TO_PROCESS > 0 and rows_processed_this_run >= NUM_ROWS_TO_PROCESS:
            logging.info(f"Reached processing limit ({NUM_ROWS_TO_PROCESS}) for this run.")
            break # Stop processing more rows

        # --- Check if row index was already processed ---
        if index in processed_row_indices:
            rows_skipped_already_done += 1
            continue # Skip this row

        # --- Process the row (if not skipped) ---
        query = row[QUERY_COLUMN_NAME]
        if pd.isna(query) or not isinstance(query, str) or not query.strip():
            logging.warning(f"Skipping row index {index} due to empty or invalid query.")
            rows_skipped_invalid_query += 1
            # Decide whether to mark invalid rows as "processed" to avoid re-checking them
            # processed_row_indices.add(index) # Optional: uncomment to skip checking invalid rows next time
            continue

        logging.info(f"Processing row index {index}, Query: '{str(query)[:100]}...'")
        answer = get_rca_for_query(str(query))

        # Add result to the list for this run
        new_results.append({
            QUERY_COLUMN_NAME: query,
            ANSWER_COLUMN_NAME: answer
        })
        # Mark this row index as processed for future runs
        processed_row_indices.add(index)
        rows_processed_this_run += 1

        # Save tracking file frequently (e.g., after each processed row)
        try:
            with open(PROCESSED_ROWS_TRACKING_FILE, "w", encoding='utf-8') as f:
                json.dump(list(processed_row_indices), f)
        except Exception as e:
            logging.error(f"Error saving tracking file after processing row index {index}: {e}")
            # Decide if you want to stop the whole process on tracking file error

        # --- Rate Limiting Delay ---
        # Add delay only if we are going to process more rows in this loop run
        # Check if it's not the last item overall AND we haven't hit the processing limit yet
        if index < len(df_input) - 1 and (NUM_ROWS_TO_PROCESS <= 0 or rows_processed_this_run < NUM_ROWS_TO_PROCESS):
             # logging.info(f"Waiting {DELAY_BETWEEN_QUERIES} seconds...") # Reduce log noise
             time.sleep(DELAY_BETWEEN_QUERIES)

    # --- Combine and Save Results ---
    if not new_results:
        logging.info("No new rows processed in this run.")
        return

    df_new_output = pd.DataFrame(new_results)
    # Append new results to existing ones (if any)
    # Use ignore_index=True to reset the DataFrame index in the combined output
    df_combined_output = pd.concat([df_output_existing, df_new_output], ignore_index=True)

    # Remove potential duplicate rows based on the query column, keeping the last entry
    # This handles cases where a row might have been processed partially before but added to tracking
    initial_rows = len(df_combined_output)
    df_combined_output.drop_duplicates(subset=[QUERY_COLUMN_NAME], keep='last', inplace=True)
    duplicates_removed = initial_rows - len(df_combined_output)
    if duplicates_removed > 0:
        logging.warning(f"Removed {duplicates_removed} duplicate rows based on '{QUERY_COLUMN_NAME}'.")


    logging.info(f"Attempting to save {len(df_combined_output)} total rows to {OUTPUT_CSV_PATH}")
    try:
        output_dir = os.path.dirname(OUTPUT_CSV_PATH)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        df_combined_output.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')

        if os.path.exists(OUTPUT_CSV_PATH): logging.info(f"Successfully saved results to {OUTPUT_CSV_PATH}")
        else: logging.error(f"Failed to save data to {OUTPUT_CSV_PATH}. File not found after write attempt.")

    except Exception as e:
        logging.error(f"Error saving output CSV to {OUTPUT_CSV_PATH}: {e}", exc_info=True)

    logging.info(f"Run Summary: Processed {rows_processed_this_run} new rows. Skipped {rows_skipped_already_done} already processed rows. Skipped {rows_skipped_invalid_query} invalid query rows.")

# --- Main Execution Guard ---
if __name__ == "__main__":
    if retriever is None:
        logging.critical("Retriever failed initialization. Cannot start batch processing.")
    else:
        process_csv()
# batch_rca_from_csv.py
import pandas as pd
import logging
import time
import os
from tqdm import tqdm
import json # Needed for tracking file
import re   # For regex pattern matching

# Import necessary components from your existing modules
from retriever import get_retriever
from prompt_llm import build_prompt, get_llm_response
from typing import List, Dict, Any

# --- Configuration ---
INPUT_CSV_PATH = "backend/filtered_25_per_category.csv" # Your input CSV
OUTPUT_CSV_PATH = "output_rca_results_batched.csv" # New output file name
QUERY_COLUMN_NAME = "heading.eng"           # Column with the actual queries
CATEGORY_COLUMN_NAME = "Categories"         # New column to include in output
ANSWER_COLUMN_NAME = "RCA_Analysis"         # Column for LLM answers

# Filter out rows where QUERY_COLUMN_NAME matches these patterns (case-insensitive)
QUERY_FILTER_PATTERNS = [
    r"### SENSOR \(INSTRUMENTAL\);MECHANICAL ###",
    r"### PROCESS ###",
    r"### SENSOR (INSTRUMENTAL) ###",
    r"### SENSOR (INSTRUMENTAL);PROCESS ###",
    r"### SENSOR (INSTRUMENTAL);MECHANICAL;CHEMICAL ###",
    r"### OTHER ###",
    r"### MECHANICAL;PROCESS ###",
    r"### MECHANICAL ###",
    r"### SENSOR (INSTRUMENTAL);CONTROL (AUTOMATION) ###",
    r"### MECHANICAL;CONTROL (AUTOMATION) ###",
    r"### MECHANICAL;PROCESS;CONTROL (AUTOMATION) ###",
    r"### SENSOR (INSTRUMENTAL);MECHANICAL;PROCESS ###",
    r"### SENSOR (INSTRUMENTAL);MECHANICAL;PROCESS;CONTROL (AUTOMATION) ###",
    r"### SENSOR (INSTRUMENTAL);CHEMICAL ###",
    r"### SENSOR (INSTRUMENTAL);MECHANICAL;CONTROL (AUTOMATION) ###",
    r"### CHEMICAL ###",
    r"### SENSOR (INSTRUMENTAL);PROCESS;CHEMICAL ###",
    r"### MECHANICAL;CHEMICAL ###",
    r"### MECHANICAL;PROCESS;CHEMICAL ###"


    # Add more patterns here if needed to skip other non-query rows
]

NUM_NEW_ROWS_PER_RUN = 100 # How many *new* rows to attempt in one script execution
OUTPUT_BATCH_SIZE = 10     # How often to save the output CSV (number of processed queries)
DELAY_BETWEEN_LLM_CALLS = 2.0 # Seconds to wait (CRUCIAL for rate limits)

# Tracking file for processed input CSV row indices
PROCESSED_ROWS_TRACKING_FILE = "backend/processed_input_csv_rows.json" # Use a clear name

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Core Components ---
retriever = None
try:
    retriever = get_retriever(k_results=7) # Increased k_results slightly, tune as needed
    logging.info("Retriever initialized successfully.")
except Exception as e:
    logging.error(f"CRITICAL: Failed to initialize retriever: {e}", exc_info=True); exit(1)

# --- Load Processed Row Indices ---
processed_row_indices = set()
if os.path.exists(PROCESSED_ROWS_TRACKING_FILE):
    try:
        with open(PROCESSED_ROWS_TRACKING_FILE, "r", encoding='utf-8') as f:
            indices_list = json.load(f)
            if isinstance(indices_list, list): processed_row_indices = set(indices_list)
            else: logging.warning(f"Format error in {PROCESSED_ROWS_TRACKING_FILE}. Starting fresh.")
        logging.info(f"Loaded {len(processed_row_indices)} already processed row indices from {PROCESSED_ROWS_TRACKING_FILE}.")
    except Exception as e:
        logging.error(f"Error loading {PROCESSED_ROWS_TRACKING_FILE}: {e}. Starting fresh.")
else:
    logging.info(f"No processed row tracking file ({PROCESSED_ROWS_TRACKING_FILE}) found. Starting fresh.")


# --- get_rca_for_query function (keep as before, handles RAG pipeline for one query) ---
def get_rca_for_query(query: str) -> str:
    if not query or not isinstance(query, str) or not query.strip(): return "ERROR: Invalid query provided."
    context_chunks_for_prompt: List[Dict[str, Any]] = []
    try: # Retrieval
        retrieved_docs = retriever.invoke(query)
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                text_content = getattr(doc, 'page_content', None); metadata = getattr(doc, 'metadata', {});
                if not isinstance(metadata, dict): metadata = {}
                chunk_id = metadata.get('id', f"retrieved_doc_{i}")
                if text_content and isinstance(text_content, str):
                    context_chunks_for_prompt.append({"id": chunk_id, "text_to_embed": text_content, "metadata": metadata})
    except Exception as e: logging.error(f"Error during retrieval for query '{query[:50]}...': {e}"); return f"ERROR: Failed during context retrieval - {e}"
    try: # Prompt Building
        final_prompt = build_prompt(query, context_chunks_for_prompt)
    except Exception as e: logging.error(f"Error building prompt for query '{query[:50]}...': {e}"); return f"ERROR: Failed during prompt building - {e}"
    try: # LLM Call
        answer = get_llm_response(final_prompt)
        return answer
    except Exception as e: logging.error(f"Error getting LLM response for query '{query[:50]}...': {e}"); return f"ERROR: Failed during LLM call - {e}"


def process_csv():
    """Loads CSV, processes rows NOT already processed, appends results in batches."""
    # --- Load Input CSV ---
    if not os.path.exists(INPUT_CSV_PATH): logging.error(f"Input CSV not found: {INPUT_CSV_PATH}"); return
    try:
        df_input = pd.read_csv(INPUT_CSV_PATH, dtype=str).fillna('') # Read all as string, fill NA
        logging.info(f"Loaded {len(df_input)} rows from {INPUT_CSV_PATH}")
    except Exception as e: logging.error(f"Error loading input CSV: {e}"); return

    # --- Check if Necessary Columns Exist ---
    required_cols = [QUERY_COLUMN_NAME, CATEGORY_COLUMN_NAME]
    for col in required_cols:
        if col not in df_input.columns:
            logging.error(f"Required column '{col}' not found in the input CSV.")
            logging.error(f"Available columns are: {list(df_input.columns)}")
            return

    # --- Read Existing Output (to append) or create empty DataFrame ---
    output_columns = [QUERY_COLUMN_NAME, CATEGORY_COLUMN_NAME, ANSWER_COLUMN_NAME]
    if os.path.exists(OUTPUT_CSV_PATH):
        try:
            df_output_existing = pd.read_csv(OUTPUT_CSV_PATH, dtype=str).fillna('')
            logging.info(f"Loaded {len(df_output_existing)} existing results from {OUTPUT_CSV_PATH}")
            # Ensure existing output has the necessary columns
            for col in output_columns:
                if col not in df_output_existing.columns:
                    df_output_existing[col] = '' # Add missing column
        except Exception as e:
            logging.warning(f"Could not read existing output file {OUTPUT_CSV_PATH}: {e}. Will create a new one.")
            df_output_existing = pd.DataFrame(columns=output_columns)
    else:
        logging.info(f"Output file {OUTPUT_CSV_PATH} not found. Creating new one.")
        df_output_existing = pd.DataFrame(columns=output_columns)

    # --- Process Rows ---
    current_batch_results = [] # Store results for the current output batch
    rows_processed_this_run = 0
    rows_skipped_already_done = 0
    rows_skipped_filtered_query = 0
    rows_skipped_invalid_query = 0

    logging.info(f"Scanning input CSV to process up to {NUM_NEW_ROWS_PER_RUN} new rows...")

    for index, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Scanning Input Rows"):
        if NUM_NEW_ROWS_PER_RUN > 0 and rows_processed_this_run >= NUM_NEW_ROWS_PER_RUN:
            logging.info(f"Reached processing limit ({NUM_NEW_ROWS_PER_RUN}) for this run.")
            break

        if index in processed_row_indices:
            rows_skipped_already_done += 1
            continue

        query = str(row[QUERY_COLUMN_NAME]).strip()
        category = str(row[CATEGORY_COLUMN_NAME]).strip()

        # Filter out queries based on patterns
        skip_due_to_pattern = False
        for pattern in QUERY_FILTER_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logging.info(f"Skipping row index {index} due to query pattern match: '{pattern}' on query '{query[:100]}...'")
                rows_skipped_filtered_query += 1
                skip_due_to_pattern = True
                processed_row_indices.add(index) # Mark as processed so we don't check it again
                break
        if skip_due_to_pattern:
            continue

        if not query: # Also catches empty strings after strip
            logging.warning(f"Skipping row index {index} due to empty query after filtering.")
            rows_skipped_invalid_query += 1
            processed_row_indices.add(index) # Mark as processed
            continue

        logging.info(f"Processing row index {index}, Query: '{query[:100]}...'")
        answer = get_rca_for_query(query)

        current_batch_results.append({
            QUERY_COLUMN_NAME: query,
            CATEGORY_COLUMN_NAME: category,
            ANSWER_COLUMN_NAME: answer
        })
        processed_row_indices.add(index)
        rows_processed_this_run += 1

        # Save tracking file frequently
        try:
            with open(PROCESSED_ROWS_TRACKING_FILE, "w", encoding='utf-8') as f:
                json.dump(list(processed_row_indices), f)
        except Exception as e:
            logging.error(f"Error saving tracking file after processing row index {index}: {e}")

        # Save output in batches
        if len(current_batch_results) >= OUTPUT_BATCH_SIZE or \
           (NUM_NEW_ROWS_PER_RUN > 0 and rows_processed_this_run == NUM_NEW_ROWS_PER_RUN) or \
           (index == len(df_input) - 1 and current_batch_results): # Save if last row and batch has data

            df_new_batch_output = pd.DataFrame(current_batch_results)
            df_output_existing = pd.concat([df_output_existing, df_new_batch_output], ignore_index=True)
            # Deduplicate based on query, keeping the latest entry for that query
            df_output_existing.drop_duplicates(subset=[QUERY_COLUMN_NAME], keep='last', inplace=True)

            try:
                output_dir = os.path.dirname(OUTPUT_CSV_PATH)
                if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
                df_output_existing.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
                logging.info(f"Saved batch to {OUTPUT_CSV_PATH}. Total rows in file: {len(df_output_existing)}")
                current_batch_results = [] # Reset for next batch
            except Exception as e:
                logging.error(f"Error saving output CSV batch: {e}", exc_info=True)

        # Rate Limiting Delay (if not the absolute last item being processed)
        if rows_processed_this_run < NUM_NEW_ROWS_PER_RUN and index < len(df_input) -1 :
            logging.debug(f"Waiting {DELAY_BETWEEN_LLM_CALLS} seconds...")
            time.sleep(DELAY_BETWEEN_LLM_CALLS)


    # Final save if any remaining items in current_batch_results not covered by loop logic
    if current_batch_results:
        df_new_batch_output = pd.DataFrame(current_batch_results)
        df_output_existing = pd.concat([df_output_existing, df_new_batch_output], ignore_index=True)
        df_output_existing.drop_duplicates(subset=[QUERY_COLUMN_NAME], keep='last', inplace=True)
        try:
            df_output_existing.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
            logging.info(f"Saved final batch to {OUTPUT_CSV_PATH}. Total rows in file: {len(df_output_existing)}")
        except Exception as e:
            logging.error(f"Error saving final output CSV batch: {e}", exc_info=True)


    logging.info(f"\n--- Run Summary ---")
    logging.info(f"Attempted to process up to {NUM_NEW_ROWS_PER_RUN} new rows.")
    logging.info(f"Actually processed and got LLM responses for: {rows_processed_this_run} new rows.")
    logging.info(f"Skipped (already processed in previous runs): {rows_skipped_already_done} rows.")
    logging.info(f"Skipped (query matched filter patterns): {rows_skipped_filtered_query} rows.")
    logging.info(f"Skipped (invalid/empty query): {rows_skipped_invalid_query} rows.")
    logging.info(f"Total rows now in output CSV {OUTPUT_CSV_PATH}: {len(df_output_existing) if 'df_output_existing' in locals() else 'Not loaded/created'}")
    logging.info(f"Total rows marked as processed in {PROCESSED_ROWS_TRACKING_FILE}: {len(processed_row_indices)}")

# --- Main Execution Guard ---
if __name__ == "__main__":
    if retriever is None:
        logging.critical("Retriever failed initialization. Cannot start batch processing.")
    else:
        process_csv()
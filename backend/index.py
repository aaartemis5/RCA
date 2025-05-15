# backend/index.py
import json
import os
# Removed: from sentence_transformers import SentenceTransformer
# <<<--- ADD OpenAI client --- >>>
import openai
# <<<------------------------ >>>
from pinecone import Pinecone
# Use updated config variables
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME, # Use alias from config
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL # Use specific OpenAI model var
)
import logging
import time
import hashlib
from typing import List, Dict, Any # For batch type hinting

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
METADATA_SIZE_LIMIT_BYTES = 35 * 1024
INPUT_DATA_PATH = "combined_output.json" # Path to combined/cleaned data
PROCESSED_IDS_FILE = "backend\processed_chunk_ids.json" # Tracking file using alias name
OPENAI_EMBEDDING_BATCH_SIZE = 100 # How many texts to send to OpenAI API at once
PINECONE_UPSERT_BATCH_SIZE = 100 # How many vectors to send to Pinecone at once

# -------------------- Initialize Pinecone --------------------
pc = None
index = None
try:
    logging.info(f"Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logging.info("Fetching list of existing Pinecone indexes...")
    index_list_response = pc.list_indexes()
    if hasattr(index_list_response, 'indexes') and isinstance(index_list_response.indexes, list):
        existing_index_names = [getattr(idx_details, 'name', None) for idx_details in index_list_response.indexes]
        existing_index_names = [name for name in existing_index_names if name is not None]
        logging.info(f"Found index names: {existing_index_names}")
        if PINECONE_INDEX_NAME not in existing_index_names:
            logging.error(f"Pinecone index '{PINECONE_INDEX_NAME}' (for OpenAI) does not exist. Please create it with the correct dimension (e.g., 1536 for text-embedding-3-small).")
            exit()
        else:
            logging.info(f"Index '{PINECONE_INDEX_NAME}' found.")
            index = pc.Index(PINECONE_INDEX_NAME)
            logging.info(f"Successfully connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
            logging.info(f"Initial index stats: {index.describe_index_stats()}")
    else:
        logging.error("Could not verify index existence. Unexpected response from pc.list_indexes().")
        exit()
except Exception as e:
    logging.exception(f"FATAL: Error during Pinecone initialization or index check: {e}")
    exit()

# -------------------- Initialize OpenAI Client --------------------
try:
    if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY not found in config.")
    # The client automatically uses OPENAI_API_KEY env var if set
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # Optional: Make a test call like client.models.list()
    logging.info("OpenAI client configured successfully.")
except Exception as e:
    logging.error(f"Error initializing OpenAI client: {e}")
    exit()

# -------------------- Load Processed Chunk IDs --------------------
processed_chunk_ids = set()
if os.path.exists(PROCESSED_IDS_FILE):
    try:
        with open(PROCESSED_IDS_FILE, "r", encoding='utf-8') as f:
            processed_ids_list = json.load(f)
            if isinstance(processed_ids_list, list): processed_chunk_ids = set(processed_ids_list)
            else: logging.warning(f"Format error in {PROCESSED_IDS_FILE}. Starting fresh.")
        logging.info(f"Loaded {len(processed_chunk_ids)} processed IDs from {PROCESSED_IDS_FILE}.")
    except Exception as e:
        logging.error(f"Error loading {PROCESSED_IDS_FILE}: {e}. Starting fresh.")
else:
    logging.info(f"No processed IDs file ({PROCESSED_IDS_FILE}) found.")

# -------------------- Load Combined JSON Data --------------------
if not os.path.exists(INPUT_DATA_PATH): logging.error(f"Error: Input data file '{INPUT_DATA_PATH}' not found."); exit()
try:
    with open(INPUT_DATA_PATH, "r", encoding='utf-8') as f: combined_data = json.load(f)
    logging.info(f"Loaded {len(combined_data)} records from {INPUT_DATA_PATH}.")
except Exception as e: logging.error(f"Error reading/decoding {INPUT_DATA_PATH}: {e}"); exit()
if not isinstance(combined_data, list): logging.error(f"Error: Expected a list of records in {INPUT_DATA_PATH}."); exit()

# --- Function to get OpenAI Embeddings in Batches ---
def get_openai_embeddings(texts: List[str], model_name: str) -> List[List[float]]:
    """Generates embeddings for a list of texts using OpenAI API."""
    if not texts: return []
    try:
        response = openai_client.embeddings.create(input=texts, model=model_name)
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        logging.error(f"Error getting OpenAI embeddings for batch: {e}")
        return [] # Return empty on error

# -------------------- Process Records --------------------
pinecone_batch = [] # Batch for upserting to Pinecone
openai_batch_texts = [] # Texts for current OpenAI API call
openai_batch_ids = []   # Corresponding IDs
openai_batch_metadata = [] # Corresponding metadata dicts

# Counters
total_chunks_processed_this_run = 0
total_skipped_already_processed = 0
total_skipped_invalid_format = 0
total_skipped_metadata_size = 0
total_skipped_embedding_error = 0
format1_count = 0; format2_count = 0

logging.info("Processing records, generating OpenAI embeddings...")

for i, record_data in enumerate(combined_data):
    chunk_id = None; text_to_embed = None; metadata_orig = None

    # --- Format Detection ---
    # (Same logic as before)
    if isinstance(record_data, dict):
        if "id" in record_data and "text_to_embed" in record_data and "metadata" in record_data and isinstance(record_data["metadata"], dict):
            chunk_id = record_data.get("id"); text_to_embed = record_data.get("text_to_embed"); metadata_orig = record_data.get("metadata"); format1_count += 1
        elif "text_chunk" in record_data and "source_document" in record_data:
            text_to_embed = record_data.get("text_chunk")
            if text_to_embed and isinstance(text_to_embed, str):
                try: hasher = hashlib.sha256(); hasher.update(text_to_embed.encode('utf-8')); chunk_id = hasher.hexdigest()
                except Exception as e: logging.error(f"Hash ID error idx {i}: {e}. Skip."); total_skipped_invalid_format += 1; continue
                metadata_orig = {"source_document": record_data.get("source_document", "N/A"), "page_number": record_data.get("start_page"), "heading": None, "timestamp": None }
                format2_count += 1
            else: logging.warning(f"Skip idx {i} (Format 2) invalid 'text_chunk'."); total_skipped_invalid_format +=1; continue
        else: logging.warning(f"Skip idx {i} unknown format. Keys: {list(record_data.keys())}"); total_skipped_invalid_format += 1; continue
    else: logging.warning(f"Skip idx {i} not a dictionary."); total_skipped_invalid_format += 1; continue

    if not chunk_id or not text_to_embed or metadata_orig is None: logging.warning(f"Skip idx {i} after format detection: missing essential data. ID: {chunk_id}"); total_skipped_invalid_format += 1; continue
    if chunk_id in processed_chunk_ids: total_skipped_already_processed += 1; continue

    # --- Prepare Final Metadata (Clean Nones, Add chunk_text) ---
    metadata_cleaned = {k: v for k, v in metadata_orig.items() if v is not None}
    metadata_cleaned["chunk_text"] = text_to_embed # Ensure text is present for retriever

    # --- Metadata Size Check ---
    try:
        metadata_json_string = json.dumps(metadata_cleaned)
        metadata_size = len(metadata_json_string.encode('utf-8'))
        if metadata_size > METADATA_SIZE_LIMIT_BYTES:
            logging.warning(f"Chunk {chunk_id} metadata size ({metadata_size}) > limit ({METADATA_SIZE_LIMIT_BYTES}) with full text. Attempting truncation.")
            temp_meta = {k:v for k,v in metadata_cleaned.items() if k != 'chunk_text'}; overhead = len(json.dumps(temp_meta).encode('utf-8')) + 50
            allowed_text_bytes = METADATA_SIZE_LIMIT_BYTES - overhead
            if allowed_text_bytes < 100: logging.error(f"Chunk {chunk_id} metadata overhead too high. Skipping."); total_skipped_metadata_size += 1; continue
            else:
                truncated_text = text_to_embed.encode('utf-8')[:allowed_text_bytes].decode('utf-8', errors='ignore') + "..."; metadata_cleaned["chunk_text"] = truncated_text
                metadata_json_string = json.dumps(metadata_cleaned); metadata_size = len(metadata_json_string.encode('utf-8')) # Recheck size
                if metadata_size > METADATA_SIZE_LIMIT_BYTES: logging.error(f"Chunk {chunk_id} metadata STILL too large after truncation. Skipping."); total_skipped_metadata_size += 1; continue
                logging.info(f"Truncated 'chunk_text' for {chunk_id}.")
    except Exception as e: logging.error(f"Metadata check/cleanup error chunk {chunk_id}: {e}. Skipping."); total_skipped_metadata_size += 1; continue

    # --- Add to Batch for OpenAI Embedding ---
    openai_batch_texts.append(text_to_embed)
    openai_batch_ids.append(chunk_id)
    openai_batch_metadata.append(metadata_cleaned)

    # --- Generate Embeddings & Prepare Pinecone Batch when OpenAI batch is full ---
    if len(openai_batch_texts) >= OPENAI_EMBEDDING_BATCH_SIZE:
        logging.info(f"Generating OpenAI embeddings for batch of {len(openai_batch_texts)} texts...")
        embeddings_list = get_openai_embeddings(openai_batch_texts, OPENAI_EMBEDDING_MODEL)

        if len(embeddings_list) == len(openai_batch_texts): # Check if embedding succeeded for all
            for chunk_id_emb, embedding, metadata_emb in zip(openai_batch_ids, embeddings_list, openai_batch_metadata):
                pinecone_batch.append((chunk_id_emb, embedding, metadata_emb)) # Add to Pinecone batch
                total_chunks_processed_this_run += 1 # Count successful embeddings
        else:
            logging.error(f"OpenAI embedding failed/incomplete for batch starting {openai_batch_ids[0]}. Skipping {len(openai_batch_texts)} chunks.")
            total_skipped_embedding_error += len(openai_batch_texts)

        # Clear OpenAI batch lists
        openai_batch_texts = []; openai_batch_ids = []; openai_batch_metadata = []

        # --- Upsert to Pinecone if *its* batch is full ---
        if len(pinecone_batch) >= PINECONE_UPSERT_BATCH_SIZE:
            try:
                logging.info(f"Upserting Pinecone batch of {len(pinecone_batch)} vectors (ends with {pinecone_batch[-1][0]})...");
                upsert_response = index.upsert(vectors=pinecone_batch)
                logging.info(f"Batch upsert SUCCEEDED. Response: {upsert_response}");
                # Update tracking file ONLY with IDs successfully prepared for upsert in this batch
                processed_chunk_ids.update([vec[0] for vec in pinecone_batch])
                with open(PROCESSED_IDS_FILE, "w", encoding='utf-8') as f: json.dump(list(processed_chunk_ids), f)
                logging.info(f"Saved IDs. Total tracked: {len(processed_chunk_ids)}")
            except Exception as e:
                failed_ids_str = ", ".join([vec[0] for vec in pinecone_batch])
                logging.error(f"Pinecone batch upsert FAILED. Error: {e}. Failed IDs: {failed_ids_str[:1000]}...");
                logging.warning(f"IDs from failed batch ({len(pinecone_batch)}) were NOT tracked.")
            finally:
                pinecone_batch = [] # Clear Pinecone batch

# --- Process remaining texts in OpenAI batch ---
if openai_batch_texts:
    logging.info(f"Generating OpenAI embeddings for final batch of {len(openai_batch_texts)} texts...")
    embeddings_list = get_openai_embeddings(openai_batch_texts, OPENAI_EMBEDDING_MODEL)
    if len(embeddings_list) == len(openai_batch_texts):
        for chunk_id_emb, embedding, metadata_emb in zip(openai_batch_ids, embeddings_list, openai_batch_metadata):
            pinecone_batch.append((chunk_id_emb, embedding, metadata_emb))
            total_chunks_processed_this_run += 1
    else:
        logging.error(f"OpenAI embedding failed/incomplete for final batch. Skipping {len(openai_batch_texts)} chunks.")
        total_skipped_embedding_error += len(openai_batch_texts)

# --- Upsert final Pinecone batch ---
if pinecone_batch:
    try:
        logging.info(f"Upserting final Pinecone batch of {len(pinecone_batch)} vectors...");
        upsert_response = index.upsert(vectors=pinecone_batch)
        logging.info(f"Final batch upsert SUCCEEDED. Response: {upsert_response}");
        processed_chunk_ids.update([vec[0] for vec in pinecone_batch])
        with open(PROCESSED_IDS_FILE, "w", encoding='utf-8') as f: json.dump(list(processed_chunk_ids), f)
    except Exception as e:
        failed_ids_str = ", ".join([vec[0] for vec in pinecone_batch])
        logging.error(f"Final batch upsert FAILED. Error: {e}. Failed IDs: {failed_ids_str[:1000]}...");
        logging.warning(f"IDs from final failed batch ({len(pinecone_batch)}) were NOT tracked.")

# --- Indexing Summary ---
logging.info(f"\n--- Indexing Summary ---")
# ... (Summary logging as before) ...
logging.info(f"Total records loaded from {INPUT_DATA_PATH}: {len(combined_data)}")
logging.info(f"Records identified as Format 1 (Structured ID): {format1_count}")
logging.info(f"Records identified as Format 2 (Hashed ID): {format2_count}")
logging.info(f"Chunks successfully embedded in this run: {total_chunks_processed_this_run}") # Renamed for clarity
logging.info(f"Chunks skipped (already processed): {total_skipped_already_processed}")
logging.info(f"Chunks skipped (invalid/unknown format): {total_skipped_invalid_format}")
logging.info(f"Chunks skipped (metadata size limit): {total_skipped_metadata_size}")
logging.info(f"Chunks skipped (embedding error): {total_skipped_embedding_error}")
logging.info(f"Total unique chunk IDs in {PROCESSED_IDS_FILE} after run: {len(processed_chunk_ids)}")



'''
# --- Sample Query Demonstration ---
# (Sample query logic remains the same)
# --- Sample Query Demonstration ---
try:
    if index:
        logging.info("Waiting a few seconds for Pinecone index to update stats...")
        time.sleep(10)
        logging.info("Checking index stats after delay...")
        index_stats = index.describe_index_stats()
        logging.info(f"Index stats after delay: {index_stats}")
        if index_stats and index_stats.total_vector_count > 0:
            logging.info("\n--- Sample Query Demonstration (Chunks) ---")
            sample_query = "What are potential causes for valve failure?" # Example query
            logging.info(f"Sample Query: {sample_query}")

            # <<< --- FIX: Generate embedding for the sample query HERE --- >>>
            try:
                query_embedding = model.encode(sample_query).tolist()
                logging.info(f"Query Embedding Generated.")
            except Exception as embed_err:
                logging.error(f"Failed to generate embedding for sample query: {embed_err}")
                query_embedding = None
            # <<< --- END FIX --- >>>

            # Only proceed if embedding was successful
            if query_embedding:
                query_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
                logging.info("\nQuery Results (Chunks):")
                if query_results and query_results["matches"]:
                    for match in query_results["matches"]:
                        logging.info(f"\n  Chunk ID: {match['id']}, Score: {match['score']:.4f}")
                        meta = match.get("metadata", {})
                        logging.info(f"    Source: {meta.get('source_document', 'N/A')}")
                        page_num = meta.get('page_number', meta.get('start_page', 'N/A')) # Check both keys
                        logging.info(f"    Page: {page_num}")
                        logging.info(f"    Heading: {meta.get('heading', 'N/A')}")
                        # Check for the text key 'chunk_text' first, then 'text_snippet'
                        text_content = meta.get('chunk_text', meta.get('text_snippet', 'N/A')) # Check both possibilities
                        logging.info(f"    Text Content (from metadata): {text_content[:250]}...") # Show snippet
                        logging.info("-" * 20)
                else:
                    logging.info("No relevant chunks found for the sample query.")
            else:
                logging.warning("Skipping sample query because embedding failed.")

        else:
            logging.warning("\nSample Query skipped. Index may be empty or stats unavailable after waiting.")
    else:
         logging.warning("\nSample Query skipped as Pinecone index object is not valid.")
except Exception as e:
    logging.error(f"Error during sample query or describing index stats: {e}")

logging.info("Script finished.")'''
import json
import os
import logging
import re
from tqdm import tqdm # For progress bar
from typing import List, Dict, Any, Optional

# --- Configuration (Updated with User Values) ---
# !!! IMPORTANT: Set the correct starting page number for your actual content !!!
START_CONTENT_PAGE = 34

# Patterns for headings typically found in TOC/Index/Front Matter to discard
# Uses regex patterns (case-insensitive)
# Updated based on user input
BAD_HEADING_PATTERNS = [
    r"^table of contents$",
    # Add more specific patterns if you notice other unwanted headings
    # Remove r"^final draft$", if you want to KEEP chunks where 'Final Draft' was the ONLY heading text
]

# Phrase to remove from the text and heading (case-insensitive)
PHRASE_TO_REMOVE = "Final Draft"

# Input and Output file paths (Using forward slashes for better cross-platform compatibility)
INPUT_JSONL_PATH = (
    "data_parsing/book14_sewageandsewerage/"
    "manual_on_sewage_and_sewerage_treatment_engineering_chunks.jsonl"
)
OUTPUT_JSON_PATH = (
    "data_parsing/book14_sewageandsewerage/"
    "manual_on_sewage_and_sewerage_treatment_engineering_chunks_CLEANED_ORDERED.json"
)


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions for Filtering & Cleaning ---

def clean_footer_phrase(text: Optional[str], phrase: str) -> Optional[str]:
    """Removes specific phrase (case-insensitive) and cleans up surrounding whitespace."""
    if not text or not phrase or not isinstance(text, str):
        return text # Return original if None, empty, or not a string
    # Regex to find the phrase with word boundaries, case-insensitive,
    # potentially surrounded by whitespace, replace with single space, then strip.
    cleaned_text = re.sub(r"\s*\b" + re.escape(phrase) + r"\b\s*", " ", text, flags=re.IGNORECASE).strip()
    # Additional cleanup for potentially double spaces left behind
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    # Return None if the string becomes empty after cleaning, otherwise the cleaned string
    return cleaned_text if cleaned_text else None

def is_valid_page(chunk_data: dict, start_page: int) -> bool:
    """Checks if the chunk's page number is within the valid content range."""
    try:
        page_num = chunk_data.get("metadata", {}).get("page_number")
        if page_num is None: return True # Keep chunks without page numbers unless filtered otherwise
        return int(page_num) >= start_page
    except (ValueError, TypeError): return False # Discard invalid page numbers

def is_valid_heading(chunk_data: dict, bad_patterns: list) -> bool:
    """Checks if the chunk's heading (already cleaned) matches any known bad patterns."""
    try:
        # Check the ALREADY CLEANED heading from the metadata
        heading = chunk_data.get("metadata", {}).get("heading")
        if heading is None: # No heading after cleaning is considered valid here
            return True

        heading_lower = heading.strip().lower() # Should already be stripped, but belt-and-suspenders
        if not heading_lower: return True # Empty heading is valid

        for pattern in bad_patterns:
            if re.match(pattern, heading_lower):
                # logging.info(f"Discarding chunk ID {chunk_data.get('id')} due to heading pattern: '{pattern}' matching cleaned heading '{heading}'")
                return False # Found a bad pattern
        return True # No bad patterns matched
    except Exception as e:
        logging.error(f"Error checking heading for chunk ID {chunk_data.get('id')}: {e}")
        return False # Discard on error

# --- Sorting Key Function ---
def chunk_sort_key(chunk_data: dict):
    """Generates a sort key tuple (page, section_idx, chunk_idx) for sorting."""
    metadata = chunk_data.get("metadata", {})
    page_num = metadata.get("page_number")
    chunk_id = chunk_data.get("id", "")
    page_sort = int(page_num) if page_num is not None else -1
    sec_idx, chunk_idx = 0, 0
    try:
        match_sec = re.search(r"_sec(\d+)_", chunk_id)
        match_chunk = re.search(r"_chunk(\d+)$", chunk_id)
        if match_sec: sec_idx = int(match_sec.group(1))
        if match_chunk: chunk_idx = int(match_chunk.group(1))
    except (TypeError, ValueError, IndexError):
        sec_idx, chunk_idx = float('inf'), float('inf') # Sort problematic IDs last
    return (page_sort, sec_idx, chunk_idx)

# --- Main Filtering, Cleaning, Reformatting, and Sorting Logic ---

def process_clean_sort_jsonl(input_path: str, output_path: str, start_page: int, bad_heading_patterns: list, phrase_to_remove: str):
    """Reads JSONL, cleans text AND headings, filters, sorts, writes pretty JSON."""
    if not os.path.isfile(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    valid_chunks = []
    total_lines_read = 0
    lines_discarded_page = 0
    lines_discarded_heading = 0
    lines_cleaned_text_phrase = 0
    lines_cleaned_heading_phrase = 0

    logging.info(f"Starting final cleaning process for: {input_path}")
    logging.info(f"Removing phrase '{phrase_to_remove}' (case-insensitive) from 'text_to_embed' AND 'metadata.heading'")
    logging.info(f"Discarding chunks before page {start_page}")
    logging.info(f"Discarding chunks with cleaned headings matching patterns: {bad_heading_patterns}")

    try:
        # Estimate total lines for tqdm
        try:
            with open(input_path, 'r', encoding='utf-8') as f_count: line_count = sum(1 for _ in f_count)
        except Exception: line_count = None

        with open(input_path, 'r', encoding='utf-8') as infile:
            for line in tqdm(infile, total=line_count, desc="Cleaning & Filtering"):
                total_lines_read += 1
                try:
                    chunk_data = json.loads(line.strip())

                    # 1. Clean the 'Final Draft' phrase from text_to_embed
                    original_text = chunk_data.get("text_to_embed", "")
                    cleaned_text = clean_footer_phrase(original_text, phrase_to_remove)
                    if original_text != cleaned_text: lines_cleaned_text_phrase += 1
                    chunk_data["text_to_embed"] = cleaned_text

                    # 2. Clean the 'Final Draft' phrase from metadata.heading
                    metadata = chunk_data.get("metadata", {})
                    original_heading = metadata.get("heading")
                    cleaned_heading = clean_footer_phrase(original_heading, phrase_to_remove)
                    if original_heading != cleaned_heading: lines_cleaned_heading_phrase += 1
                    # Update the metadata with the cleaned heading (or None if it became empty)
                    metadata["heading"] = cleaned_heading
                    chunk_data["metadata"] = metadata # Ensure metadata is updated in chunk_data

                    # 3. Apply filters (Page Number) - Uses original chunk_data
                    if not is_valid_page(chunk_data, start_page):
                        lines_discarded_page += 1
                        continue

                    # 4. Apply filters (Heading) - Uses chunk_data with the *cleaned* heading
                    if not is_valid_heading(chunk_data, bad_heading_patterns):
                        lines_discarded_heading += 1
                        continue

                    # Optional: Discard if cleaning made text too short?
                    # MIN_CHUNK_LENGTH = 50 # Define this in config
                    # if len(cleaned_text) < MIN_CHUNK_LENGTH:
                    #     continue

                    # If all checks pass, add the *modified* chunk to the list
                    valid_chunks.append(chunk_data)

                except json.JSONDecodeError: logging.warning(f"Skipping invalid JSON line {total_lines_read}")
                except Exception as e: logging.error(f"Error processing line {total_lines_read}: {e}")

    except Exception as e:
        logging.error(f"An error occurred during file reading: {e}", exc_info=True)
        return

    logging.info(f"Read {total_lines_read} lines.")
    logging.info(f"Removed '{phrase_to_remove}' from text in {lines_cleaned_text_phrase} chunks.")
    logging.info(f"Removed '{phrase_to_remove}' from heading in {lines_cleaned_heading_phrase} chunks.")
    logging.info(f"Discarded {lines_discarded_page} lines (Page < {start_page}).")
    logging.info(f"Discarded {lines_discarded_heading} lines (Bad Heading pattern matched on cleaned heading).")
    logging.info(f"Keeping {len(valid_chunks)} chunks.")

    # 5. Sort the valid chunks
    logging.info(f"Sorting {len(valid_chunks)} chunks...")
    try:
        valid_chunks.sort(key=chunk_sort_key)
        logging.info("Sorting complete.")
    except Exception as e:
        logging.error(f"An error occurred during sorting: {e}", exc_info=True)
        logging.warning("Proceeding with potentially unsorted data.")

    # 6. Write the final, pretty JSON file
    logging.info(f"Writing {len(valid_chunks)} final chunks to: {output_path}")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(valid_chunks, outfile, indent=4, ensure_ascii=False) # Pretty print!
        logging.info("Successfully wrote final cleaned, ordered, and formatted JSON file.")
    except Exception as e:
        logging.error(f"Failed to write final output JSON file: {e}", exc_info=True)

# --- Script Execution ---
if __name__ == "__main__":
    process_clean_sort_jsonl(
        INPUT_JSONL_PATH,
        OUTPUT_JSON_PATH,
        START_CONTENT_PAGE,
        BAD_HEADING_PATTERNS,
        PHRASE_TO_REMOVE
    )
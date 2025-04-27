###run this code in google colab wont load unstructured here 


import os
import json
import re
import logging
import gc
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean

# === Configuration ===
PDF_STRATEGY = "hi_res"         # PDF parsing quality: 'hi_res' or 'fast'
CHUNK_SIZE = 1000                # Maximum characters per text chunk
CHUNK_OVERLAP = 150              # Overlap between consecutive chunks
INCLUDE_ELEMENT_TYPES = {        # Elements to retain for content extraction
    'NarrativeText', 'ListItem', 'Table'
}
DISCARD_ELEMENT_TYPES = {        # Elements to skip during parsing
    'Header', 'Footer', 'PageNumber', 'Image', 'FigureCaption'
}
MIN_CHUNK_LENGTH = 50            # Discard chunks shorter than this length
MAX_THREADS = 8                  # Number of worker threads for parallel processing
START_CONTENT_PAGE = 32          # First page of main content (inclusive)
END_CONTENT_PAGE = None          # Last content page; None to read until end

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Helper Functions ===
def clean_text(text: str) -> str:
    """
    Remove page numbers, normalize whitespace, and strip extra punctuation.
    """
    if not isinstance(text, str):
        return ""
    # Drop leading numerals and apply unstructured cleaner
    text = re.sub(r"^\s*\d+\s+", "", text, flags=re.MULTILINE)
    return clean(
        text,
        bullets=False,
        extra_whitespace=True,
        dashes=False,
        trailing_punctuation=False
    ).strip()


def split_text(text: str, splitter: RecursiveCharacterTextSplitter) -> List[str]:
    """
    Divide text into chunks using the provided splitter; fallback to full text on error.
    """
    try:
        return splitter.split_text(text)
    except Exception as e:
        logging.warning(f"Splitter error: {e}")
        return [text]


def generate_chunk_id(
    base: str,
    page: Optional[int],
    sec_idx: int,
    chunk_idx: int
) -> str:
    """
    Construct a unique identifier for each text chunk.
    """
    page_tag = page if page is not None else 'N/A'
    return f"{base}_pg{page_tag}_sec{sec_idx}_chunk{chunk_idx}"


def process_single_section(
    base_name: str,
    page: Optional[int],
    sec_idx: int,
    heading_elem: Any,
    sec_text: str,
    splitter: RecursiveCharacterTextSplitter
) -> List[Dict[str, Any]]:
    """
    Clean, split, and package one document section into embeddable chunks.
    """
    # Prepare heading and body text
    heading = clean_text(heading_elem.text) if heading_elem else "Untitled"
    body = clean_text(sec_text)
    if len(body) <= MIN_CHUNK_LENGTH:
        return []

    # Split into manageable chunks
    raw_chunks = split_text(body, splitter)
    results = []
    for idx, chunk in enumerate(raw_chunks, start=1):
        text = chunk.strip()
        if len(text) < MIN_CHUNK_LENGTH:
            continue
        chunk_id = generate_chunk_id(base_name, page, sec_idx, idx)
        results.append({
            "id": chunk_id,
            "text_to_embed": text,
            "metadata": {
                "source_document": base_name,
                "page_number": page,
                "heading": heading,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        })
    # Release memory
    del sec_text, body, raw_chunks
    gc.collect()
    return results


def process_pdf_append_output(pdf_path: str, output_jsonl_path: str):
    """
    Extract content from specified PDF pages, split into chunks, and write to JSONL.
    """
    if not os.path.isfile(pdf_path):
        logging.error(f"Missing PDF: {pdf_path}")
        return

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    logging.info(f"Parsing PDF: {base_name} (strategy={PDF_STRATEGY})")

    # Parse document pages
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy=PDF_STRATEGY,
            first_page=START_CONTENT_PAGE,
            last_page=END_CONTENT_PAGE,
            infer_table_structure=True,
            include_page_breaks=False,
            extract_images_in_pdf=False
        )
    except Exception as e:
        logging.error(f"PDF partitioning failed: {e}")
        return

    logging.info(f"Found {len(elements)} parsed elements.")

    # Configure text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Identify document sections
    logging.info("Building sections...")
    sections: List[Tuple[Optional[int], int, Any, str]] = []
    current_text, current_heading, current_page = [], None, None
    sec_idx = 0

    for elem in elements:
        typ = type(elem).__name__
        page = getattr(elem.metadata, 'page_number', None)

        # Skip unwanted types
        if typ in DISCARD_ELEMENT_TYPES:
            continue

        # Start new section on Title
        if typ == 'Title':
            if current_text:
                sec_idx += 1
                sections.append((current_page, sec_idx, current_heading, "".join(current_text)))
            current_heading = elem
            current_text, current_page = [], page
        elif typ in INCLUDE_ELEMENT_TYPES:
            if current_page is None:
                current_page = page
            text = getattr(elem, 'text', '')
            if text:
                current_text.append(text + "\n\n")

    # Add final section
    if current_text:
        sec_idx += 1
        sections.append((current_page, sec_idx, current_heading, "".join(current_text)))

    logging.info(f"Total sections: {len(sections)}")

    # Prepare output
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    if os.path.exists(output_jsonl_path):
        logging.warning(f"Overwriting {output_jsonl_path}")
        os.remove(output_jsonl_path)

    # Process sections and write chunks
    total = 0
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor, \
         open(output_jsonl_path, 'a', encoding='utf-8') as out_file:

        futures = [executor.submit(
            process_single_section,
            base_name, page, idx, heading, text, splitter
        ) for page, idx, heading, text in sections]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Sections"):
            try:
                chunks = future.result()
                for c in chunks:
                    out_file.write(json.dumps(c, ensure_ascii=False) + "\n")
                total += len(chunks)
            except Exception as e:
                logging.error(f"Section error: {e}")
            gc.collect()

    logging.info(f"Wrote {total} chunks to {output_jsonl_path}")


# === Script Entry Point ===
if __name__ == "__main__":
    INPUT_PDF = "manual_on_sewage_and_sewerage_treatment_engineering.pdf"
    OUTPUT_DIR = "output_directory"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{os.path.splitext(os.path.basename(INPUT_PDF))[0]}_chunks.jsonl"
    )
    process_pdf_append_output(INPUT_PDF, output_file)

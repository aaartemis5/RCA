import json
import os

# <<<--- ADD OpenAI client --- >>>
import openai
# <<<------------------------ >>>
from langchain.prompts import PromptTemplate
# Use updated OpenAI config variables
from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL # Use specific OpenAI vars
import logging
from typing import List, Dict, Any

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize OpenAI Client ---
openai_client = None
try:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in config.py or environment variables.")
    # Configure the client using the key from config
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # Optional: Test connection if desired (e.g., client.models.list())
    logging.info("OpenAI client initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing OpenAI client: {e}")
    # openai_client will remain None, handled in get_llm_response

# --- format_context function (No changes needed) ---
def format_context(context_chunks: list) -> str:
    """
    Formats the list of retrieved context chunks into a single string for the prompt.
    Each chunk's metadata (heading, page) is included.
    """
    formatted_string = ""
    for i, chunk_data in enumerate(context_chunks):
        # Use 'text_to_embed' key which matches the structure created by app.py
        text = chunk_data.get("text_to_embed", "N/A")
        metadata = chunk_data.get("metadata", {})
        # Extract relevant metadata fields (adjust keys if needed based on index.py)
        heading = metadata.get("heading", "N/A")
        page = metadata.get("page_number", metadata.get("start_page", "N/A")) # Check both page keys
        source = metadata.get("source_document", "N/A")

        formatted_string += f"--- Context Chunk {i+1} ---\n"
        formatted_string += f"Source: {source}\n"
        formatted_string += f"Page: {page}\n"
        formatted_string += f"Heading: {heading}\n"
        formatted_string += f"Text: {text}\n"
        formatted_string += "---------------------------\n\n"

    # Crude truncation - check limits for OPENAI_CHAT_MODEL (GPT-4 Turbo has large limits)
    MAX_CONTEXT_LENGTH = 25000 # Example limit, adjust based on model and desired output length
    if len(formatted_string) > MAX_CONTEXT_LENGTH:
        logging.warning(f"Formatted context length ({len(formatted_string)}) exceeded limit ({MAX_CONTEXT_LENGTH}). Truncating.")
        formatted_string = formatted_string[:MAX_CONTEXT_LENGTH] + "\n... [Context Truncated]"

    return formatted_string

# --- build_prompt function (No changes needed - uses your RCA instructions) ---
def build_prompt(query: str, context_chunks: list) -> str:
    """
    Builds the prompt for the LLM for STP RCA, including instructions and formatted context.
    """
    formatted_context = format_context(context_chunks)
    # Your detailed RCA-specific Prompt Template
    prompt_template = """
You are an expert AI assistant specializing in Root Cause Analysis (RCA) for Sewage Treatment Plant (STP) operations and troubleshooting. Your knowledge comes *exclusively* from the provided context snippets extracted from technical manuals.

**Goal:** Help the user identify the potential *root cause(s)* of the problem described in their query by analyzing the provided context.

**Instructions:**
1. **Analyze Query:** Understand the problem symptoms in the "User Query".
2. **Use Context ONLY:** Base your main analysis strictly on the "Context Snippets" provided. Do *not* use outside knowledge unless explicitly allowed below.
3. **Start with a Summary Section:** Begin your answer with:

    **Summary of Likely Root Cause(s):**
    - (List 2 to 4 possible root causes in 1 line each based *only* on context analysis)

4. **Then add Detailed Explanation with the following structure:**

    **Detailed Root Cause Analysis**
    - **Problem Summary:**
      (1–2 lines restating the user’s problem)
    - **Potential Immediate Causes (Based on Context):**
      (List possible direct causes mentioned or implied in context)
    - **Potential Root Causes & Reasoning (Based on Context):**
      (Explain underlying causes, linking context snippets where possible)
    - **Recommended Diagnostic Steps (Based on Context):**
      (List specific checks suggested or implied by the context)

5. **Avoid Inline Citations:** Do NOT include specific headings or page numbers within the main answer body paragraphs.
6. **Add References Section at the End:**
    - List the sources used in the analysis. Use this format for each relevant chunk:
      * `Source: [Source Document] – Heading: [Heading] – Page: [Page Number]`
7. **If Context is Insufficient:**
    - If the context clearly doesn't address the query, add a section at the end called **Insufficient Context**, clearly stating what information seems missing. Do not speculate wildly.
8. **Only Then Add a Section Called "General Knowledge (Optional)"**
    - **If and only if** the context was insufficient or to provide minor clarification, you *may* add brief, general troubleshooting points *clearly labeled* under this heading, but prioritize context-based analysis.
9. **Be concise, factual, and organized. Use a readable tone.**

Context Snippets:
---------------------
{context}
---------------------

User Query: {query}

**Root Cause Analysis:**
""" # End of prompt template string
    prompt = PromptTemplate(input_variables=["query", "context"], template=prompt_template)
    formatted_prompt = prompt.format(query=query, context=formatted_context)
    return formatted_prompt


# --- Updated function to call OpenAI LLM ---
def get_llm_response(prompt: str) -> str:
    """Sends the prompt to the configured OpenAI LLM and returns the response."""
    if not openai_client:
        logging.error("OpenAI client not initialized.")
        return "ERROR: LLM client is not available. Please check configuration and API key."

    try:
        logging.info(f"Sending prompt to OpenAI model: {OPENAI_CHAT_MODEL}")
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL, # Use the specific model from config
            messages=[
                # System message defines the core persona and constraints
                {"role": "system", "content": "You are an AI expert performing Root Cause Analysis for Sewage Treatment Plant problems based *only* on the provided text context. Structure your answer clearly, starting with a summary, then details, diagnostics, and references based ONLY on the context. State if context is insufficient."},
                # The user prompt contains the detailed instructions, query, and formatted context
                {"role": "user", "content": prompt}
            ],
            #temperature=0.5, # Low temperature for factual, consistent output
            #max_tokens=2000, # Optional: Limit OpenAI's response length if needed
        )
        # Process the response from OpenAI API v1.x+
        if response.choices and response.choices[0].message and response.choices[0].message.content:
             llm_answer = response.choices[0].message.content.strip()
             logging.info(f"Received response from OpenAI (length: {len(llm_answer)} chars).")
             # Optional: Log token usage if needed (check response object structure)
             # usage = response.usage
             # logging.info(f"OpenAI Token Usage: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}")
             return llm_answer
        else:
             logging.error("OpenAI response structure unexpected or empty.")
             return "I apologize, but I received an unexpected response format from the AI model."

    except openai.AuthenticationError as e:
        logging.error(f"OpenAI Authentication Error: {e}")
        return "ERROR: Cannot authenticate with OpenAI. Please check your API key."
    except openai.RateLimitError as e:
         logging.error(f"OpenAI Rate Limit Error: {e}")
         return "ERROR: OpenAI API rate limit exceeded. Please check your plan or wait."
    except openai.APIConnectionError as e:
         logging.error(f"OpenAI Connection Error: {e}")
         return "ERROR: Could not connect to OpenAI API. Please check your network."
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}", exc_info=True)
        return f"I'm sorry, but I encountered an unexpected error communicating with the OpenAI model."

if __name__ == "__main__":
    # Example usage for testing this module directly
    print("\n--- Testing prompt_llm.py for STP RCA ---")

    # Simulate a user query about an STP problem
    sample_query = "We are experiencing excessive white, thick foam in the aeration tank for the past day. Dissolved Oxygen (DO) is reading slightly low around 1.0 mg/L, but MLSS concentration appears normal."

    # Simulate context chunks that might be retrieved by RAG from your processed JSON
    # Each item in the list represents a chunk dictionary
    sample_context_chunks = [
        {
            "id": "manual_pg150_sec5_chunk2",
            "text_to_embed": "Foaming: Excessive foaming can occur in aeration tanks. White, billowy foam often indicates young sludge (low Mean Cell Residence Time - MCRT), particularly during startup or after a process washout event. Check sludge age calculations. Filamentous bacteria like Nocardia typically cause a thick, stable, brown scum, not white foam. Surfactants in the influent can also cause rapid-onset white foaming.",
            "metadata": { "source_document": "manual_STP.pdf", "page_number": 150, "heading": "5.8.1 Aeration Tank Operations", "timestamp": "..."}
        },
        {
            "id": "manual_pg152_sec5_chunk5",
            "text_to_embed": "Mean Cell Residence Time (MCRT): MCRT, or sludge age, is critical for process stability. Low MCRT (e.g., < 3 days in typical conventional plants) may not allow for a mature microbial population, leading to instability and potential foaming. Calculation requires influent TSS, effluent TSS, WAS flow rate, and WAS concentration. Ensure accurate flow measurements.",
            "metadata": { "source_document": "manual_STP.pdf", "page_number": 152, "heading": "5.8.1.2 Process Variables", "timestamp": "..."}
        },
        {
            "id": "manual_pg95_sec4_chunk1",
            "text_to_embed": "Dissolved Oxygen (DO): Maintaining adequate DO (typically 1.5-2.5 mg/L in conventional activated sludge) is vital for aerobic treatment. Low DO can stress the biomass and favor growth of certain filamentous organisms, although it's not the primary cause of typical white startup foam. Persistently low DO may indicate overloading (high influent BOD) or insufficient aeration capacity. Check blower output and diffuser condition.",
            "metadata": { "source_document": "manual_STP.pdf", "page_number": 95, "heading": "4.7 Aeration Systems", "timestamp": "..."}
        },
         {
            "id": "manual_pg50_sec3_chunk8",
            "text_to_embed": "Influent Characteristics: Monitor influent for unusual discharges. Sudden spikes in detergents or surfactants from industrial sources can cause rapid foaming events in the aeration basin. Visual inspection of the influent channel may reveal foam.",
            "metadata": { "source_document": "manual_STP.pdf", "page_number": 50, "heading": "3.4 Influent Monitoring", "timestamp": "..."}
        },
        {
            # ID would be the hash generated during indexing
            "id": "a1b2c3d4e5f6...", # Example placeholder hash ID
            "text_to_embed": "These include problems with equipment design, fabrication, installation, maintenance, and misuse. Prob- lems with the equipment reliability program are also identified/categorized under this node. Typical Recommendation See lower level nodes. Example A spill to the environment occurred because a valve failed. The valve failed because it was not designed for the environment in which it operated.",
            "metadata": {
                "source_document": "RCA_ABSgroup.pdf",
                "page_number": 1, # Originally 'start_page'
                "heading": None, # No heading available in original format
                "timestamp": "2025-05-01T12:00:00Z" # Example timestamp
            }
        },
        {
            "id": "f6e5d4c3b2a1...", # Example placeholder hash ID
            "text_to_embed": "These include problems related to the design process, problems related to the design and capabilities of the equipment, and problems related to the specification of parts and materials. Typical Recommendation See lower level nodes. A valve failed because the designer used obsolete materials requirements. A process upset occurred because one of the flow streams was out of specification. The design input did not indicate all the possible flow rates for the process. The pump was incorrectly sized for the necessary flow requirements. A line ruptured because a gasket failed. The gasket was constructed of the wrong material because the design did not consider all the possible chemicals that would be in the line during different operating condi- tions. A chemical that was not considered caused the gasket to fail.",
            "metadata": {
                "source_document": "RCA_ABSgroup.pdf",
                "page_number": 4, # Originally 'start_page'
                "heading": None, # No heading available in original format
                "timestamp": "2025-05-01T12:00:01Z" # Example timestamp
            }
        },
    ]

    print(f"Sample Query: {sample_query}")
    # print(f"Sample Context Chunks:\n{json.dumps(sample_context_chunks, indent=2)}") # Optional: print formatted chunks

    # Build the prompt using the list of chunk dictionaries
    full_prompt = build_prompt(sample_query, sample_context_chunks)
    # print("\nConstructed Prompt (first 500 chars):\n", full_prompt[:500], "...") # Optional: print start of prompt

    print("\nGetting LLM response...")
    answer = get_llm_response(full_prompt)
    print("\nLLM Response:\n", answer)
!pip install vllm pdf2image pypdf

# Code v1:

import json
import os
from vllm import LLM, SamplingParams
from pdf2image import convert_from_path

# --- 1. CONFIGURATION: DEFINE YOUR FIELDS HERE ---
EXTRACTION_FIELDS = {
    "unique_id": "The Clause number, SKU, or Invoice ID.",
    "vendor_name": "The name of the company or person.",
    "invoice_date": "The date the document was issued.",
    "description": "The full detailed text (STITCHED across pages).",
    "total_amount": "The final numerical value for this item."
}

# --- 2. ENGINE CONFIG ---
# Instruct models are faster and use less VRAM than Thinking models.
# We can increase parallelism (max_num_seqs) to 4 or 8.
model_id = "Qwen/Qwen3-VL-8B-Instruct"

llm = LLM(
    model=model_id,
    max_model_len=32768,      # Standard context window
    max_num_seqs=4,           # Increased parallel chunks for speed
    trust_remote_code=True,
    mm_processor_kwargs={
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 28 * 28, # High-res for OCR accuracy
    }
)

sampling_params = SamplingParams(temperature=0, max_tokens=2048)

# --- 3. DYNAMIC PROMPT GENERATION ---
fields_str = "\n".join([f"- '{k}': {v}" for k, v in EXTRACTION_FIELDS.items()])
SYSTEM_PROMPT = f"""
You are a precise document extraction agent. 
Extract the following fields into a JSON list:

{fields_str}
- 'is_complete': true if the entry ends on this page, false if it continues.

RULES:
1. UNIQUE ID: Always provide a 'unique_id'. If it spans pages, REPEAT the same ID.
2. FORMAT: Return ONLY a raw JSON list. No conversational text. Do not include markdown code blocks.
"""

def clean_and_parse_json(text):
    """Simple cleaner for Instruct model responses."""
    try:
        # Strip potential markdown if the model ignores the 'no markdown' rule
        clean = text.strip().strip("`").replace("json\n", "")
        return json.loads(clean)
    except:
        return None

def process_instruct_document(pdf_path, chunk_size=3):
    print(f"--- Starting Instruct Extraction for {pdf_path} ---")
    images = convert_from_path(pdf_path, dpi=200)
    total_pages = len(images)
    master_storage = {}
    
    # Pre-build all requests for vLLM batching
    all_requests = []
    for i in range(0, total_pages, chunk_size):
        chunk_images = images[i : i + chunk_size]
        start, end = i + 1, min(i + chunk_size, total_pages)
        
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
        prompt += "<|vision_start|><|vision_end|>" * len(chunk_images)
        prompt += f"Extract data from pages {start} to {end}.<|im_end|>\n<|im_start|>assistant\n"
        
        all_requests.append({
            "prompt": prompt,
            "multi_modal_data": {"image": chunk_images}
        })

    # Execute all chunks in parallel (Batching)
    outputs = llm.generate(all_requests, sampling_params)

    # Stitch the results
    for output in outputs:
        data = clean_and_parse_json(output.outputs[0].text)
        
        if data:
            for entry in data:
                uid = entry.get("unique_id")
                if not uid: continue
                
                if uid in master_storage:
                    # STITCHING: Combine the 'description' field
                    master_storage[uid]["description"] += " " + entry.get("description", "")
                    master_storage[uid]["is_complete"] = entry.get("is_complete", True)
                else:
                    master_storage[uid] = entry
        else:
            print(f"⚠️ JSON Parse error on a chunk. Output: {output.outputs[0].text[:100]}...")

    return list(master_storage.values())

# --- 4. RUN ---
if __name__ == "__main__":
    results = process_instruct_document("input_file.pdf")
    with open("final_data_instruct.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Extraction Finished. Total items: {len(results)}")



# Code V2:

import json
import re
import os
import time
import random
from vllm import LLM, SamplingParams
from pdf2image import convert_from_path

# --- 1. CONFIGURATION: DEFINE YOUR FIELDS HERE ---
# unique_id is REQUIRED for the stitching logic to work.
EXTRACTION_FIELDS = {
    "unique_id": "The Clause number, SKU, or Invoice ID.",
    "vendor_name": "The name of the company or person.",
    "invoice_date": "The date the document was issued.",
    "description": "The full detailed text (STITCHED across pages).",
    "total_amount": "The final numerical value for this item."
}

# --- 2. ENGINE CONFIG ---
model_id = "Qwen/Qwen3-VL-8B-Thinking"
llm = LLM(
    model=model_id,
    max_model_len=64000, 
    max_num_seqs=2, # Parallel chunks (keep low for Thinking models)
    trust_remote_code=True,
    mm_processor_kwargs={
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1024 * 28 * 28, # Standard 300 DPI quality
    }
)

sampling_params = SamplingParams(temperature=0, max_tokens=4096)

# --- 3. DYNAMIC PROMPT GENERATION ---
fields_str = "\n".join([f"- '{k}': {v}" for k, v in EXTRACTION_FIELDS.items()])
SYSTEM_PROMPT = f"""
You are a precise document extraction agent. 
First, reason step-by-step about the layout and Unique IDs. 
Then, extract the following fields into a JSON list:

{fields_str}
- 'is_complete': true if the entry ends on this page, false if it continues.

RULES:
1. UNIQUE ID: Always provide a 'unique_id'. If it spans pages, REPEAT the same ID.
2. FORMAT: Return ONLY a JSON list inside ```json blocks.
"""

def extract_thought_and_json(text):
    json_match = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    thought_part = text.split("```json")[0].strip()
    if json_match:
        try:
            return thought_part, json.loads(json_match[-1])
        except:
            return thought_part, None
    return text, None

def process_document(pdf_path, chunk_size=2):
    print(f"--- Starting Extraction for {pdf_path} ---")
    images = convert_from_path(pdf_path, dpi=200)
    total_pages = len(images)
    master_storage = {}
    
    with open("extraction_log.txt", "w", encoding="utf-8") as log:
        for i in range(0, total_pages, chunk_size):
            start, end = i + 1, min(i + chunk_size, total_pages)
            chunk_images = images[i : i + chunk_size]
            
            # Construct Multimodal Prompt
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
            prompt += "<|vision_start|><|vision_end|>" * len(chunk_images)
            prompt += f"Extract pages {start} to {end}.<|im_end|>\n<|im_start|>assistant\n"

            # Run Inference
            outputs = llm.generate([prompt], sampling_params, multi_modal_data={"image": chunk_images})
            raw_text = outputs[0].outputs[0].text

            # Parse & Log
            thought, data = extract_thought_and_json(raw_text)
            log.write(f"--- PAGES {start}-{end} ---\nREASONING:\n{thought}\n\n")
            
            if data:
                for entry in data:
                    uid = entry.get("unique_id")
                    if not uid: continue
                    
                    if uid in master_storage:
                        # STITCHING LOGIC: Combine the 'description' field
                        master_storage[uid]["description"] += " " + entry.get("description", "")
                        master_storage[uid]["is_complete"] = entry.get("is_complete", True)
                    else:
                        master_storage[uid] = entry
            else:
                log.write("!!! JSON PARSE FAILED FOR THIS CHUNK !!!\n")
            
            log.write("="*50 + "\n")

    return list(master_storage.values())

# --- 4. RUN ---
if __name__ == "__main__":
    results = process_document("your_file.pdf")
    with open("final_data.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Success! Extracted {len(results)} items. Check extraction_log.txt for AI thoughts.")

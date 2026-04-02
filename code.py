!pip install vllm pdf2image pypdf

# Code v1:

import json
import time
from vllm import LLM, SamplingParams
from pdf2image import convert_from_path

# --- 1. SETUP ENGINE ---
model_id = "Qwen/Qwen3-VL-8B-Instruct"
llm = LLM(
    model=model_id,
    max_model_len=64000, # Adjust based on your HBM size
    max_num_seqs=4,
    trust_remote_code=True,
    mm_processor_kwargs={
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 28 * 28, # Standard 300 DPI quality
    }
)

# --- 2. MASTER PROMPT ---
SYSTEM_PROMPT = """
Extract document data into a JSON list. 
RULES:
1. UNIQUE ID: Use 'unique_id' for every object (e.g., Clause 1.2).
2. CONTINUATION: If a section spans pages, REPEAT the same 'unique_id'.
3. INCOMPLETE FLAG: 'is_complete': false if text hits the bottom and continues.
4. FORMAT: Return ONLY a raw JSON list. No conversational text.
"""

def safe_parse_json(text):
    """Cleans and parses JSON from the model response."""
    try:
        clean = text.strip().strip("`").replace("json\n", "")
        return json.loads(clean)
    except Exception:
        return None

def process_with_retries(pdf_path, chunk_size=3, max_retries=3):
    images = convert_from_path(pdf_path, dpi=200)
    total_pages = len(images)
    
    # Track chunks that need processing
    pending_chunks = []
    for i in range(0, total_pages, chunk_size):
        pending_chunks.append({
            "id": i,
            "images": images[i : i + chunk_size],
            "pages": (i + 1, min(i + chunk_size, total_pages)),
            "attempt": 0
        })

    master_storage = {}
    
    while pending_chunks:
        current_requests = []
        for chunk in pending_chunks:
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
            prompt += "<|vision_start|><|vision_end|>" * len(chunk["images"])
            prompt += f"Extract pages {chunk['pages'][0]} to {chunk['pages'][1]}.<|im_end|>\n<|im_start|>assistant\n"
            
            # For retries, we can vary temperature to get a different result
            temp = 0.2 if chunk["attempt"] > 0 else 0
            current_requests.append({
                "prompt": prompt,
                "multi_modal_data": {"image": chunk["images"]},
                "params": SamplingParams(temperature=temp, max_tokens=2048)
            })

        # Process the batch in parallel
        outputs = llm.generate(
            [r["prompt"] for r in current_requests],
            [r["params"] for r in current_requests],
            use_tqdm=True
        )

        failed_chunks = []
        for i, output in enumerate(outputs):
            chunk = pending_chunks[i]
            data = safe_parse_json(output.outputs[0].text)
            
            if data is not None:
                # Merge data using Unique ID logic
                for entry in data:
                    uid = entry.get("unique_id")
                    if uid in master_storage:
                        master_storage[uid]["description"] += " " + entry.get("description", "")
                        master_storage[uid]["is_complete"] = entry.get("is_complete", True)
                    else:
                        master_storage[uid] = entry
            else:
                # Retry logic with exponential backoff
                chunk["attempt"] += 1
                if chunk["attempt"] < max_retries:
                    print(f"⚠️ Retry {chunk['attempt']} for pages {chunk['pages']}...")
                    failed_chunks.append(chunk)
                else:
                    print(f"❌ Failed permanently: pages {chunk['pages']}")

        pending_chunks = failed_chunks
        if pending_chunks:
            time.sleep(2 ** len(failed_chunks)) # Simple backoff

    return list(master_storage.values())

# --- 3. EXECUTION ---
if __name__ == "__main__":
    final_json = process_with_retries("input.pdf")


# Code V2 :

import json
import time
import random
from vllm import LLM, SamplingParams
from pdf2image import convert_from_path

# --- 1. OPTIMIZED ENGINE CONFIG ---
model_id = "Qwen/Qwen3-VL-8B-Instruct"

llm = LLM(
    model=model_id,
    max_model_len=32768,      # Balanced for multi-image chunks
    max_num_seqs=8,           # INCREASED for higher parallel throughput
    trust_remote_code=True,
    mm_processor_kwargs={
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1024 * 28 * 28, # SLIGHTLY LOWERED for faster pre-processing
    }
)

# --- 2. MASTER PROMPT ---
SYSTEM_PROMPT = """
Extract document data into a JSON list. 
RULES:
1. UNIQUE ID: Use 'unique_id' for every object.
2. CONTINUATION: If a section spans pages, REPEAT the same 'unique_id'.
3. INCOMPLETE FLAG: 'is_complete': false if text hits the bottom and continues.
4. FORMAT: Return ONLY a raw JSON list. No conversational text.
"""

def safe_parse_json(text):
    try:
        clean = text.strip().strip("`").replace("json\n", "")
        return json.loads(clean)
    except:
        return None

def process_fast_with_retries(pdf_path, chunk_size=3, max_retries=2):
    # Lower DPI (150-200) improves speed while Qwen3's dynamic res maintains detail
    images = convert_from_path(pdf_path, dpi=180) 
    total_pages = len(images)
    
    pending_chunks = []
    for i in range(0, total_pages, chunk_size):
        pending_chunks.append({
            "images": images[i : i + chunk_size],
            "pages": (i + 1, min(i + chunk_size, total_pages)),
            "attempt": 0
        })

    master_storage = {}
    
    while pending_chunks:
        current_reqs = []
        for chunk in pending_chunks:
            # Shift temperature on retries to break formatting loops
            temp = 0.3 if chunk["attempt"] > 0 else 0
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
            prompt += "<|vision_start|><|vision_end|>" * len(chunk["images"])
            prompt += f"Extract pages {chunk['pages']}.<|im_end|>\n<|im_start|>assistant\n"
            
            current_reqs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": chunk["images"]},
                "params": SamplingParams(temperature=temp, max_tokens=2048)
            })

        # BATCH INFERENCE
        outputs = llm.generate(
            [r["prompt"] for r in current_reqs],
            [r["params"] for r in current_reqs]
        )

        failed_chunks = []
        for i, output in enumerate(outputs):
            chunk = pending_chunks[i]
            data = safe_parse_json(output.outputs.text)
            
            if data:
                for entry in data:
                    uid = entry.get("unique_id")
                    if uid in master_storage:
                        master_storage[uid]["description"] += " " + entry.get("description", "")
                        master_storage[uid]["is_complete"] = entry.get("is_complete", True)
                    else:
                        master_storage[uid] = entry
            else:
                chunk["attempt"] += 1
                if chunk["attempt"] <= max_retries:
                    failed_chunks.append(chunk)

        pending_chunks = failed_chunks
        if pending_chunks:
            # Exponential Backoff with Jitter for throughput stability
            wait_time = (2 ** pending_chunks[0]["attempt"]) + random.uniform(0, 1)
            time.sleep(wait_time)

    return list(master_storage.values())

if __name__ == "__main__":
    final_output = process_fast_with_retries("input.pdf")
    with open("optimized_output.json", "w") as f:
        json.dump(final_output, f, indent=4)
    with open("output.json", "w") as f:
        json.dump(final_json, f, indent=4)


# Code V3 :

import json
import re
import os
from vllm import LLM, SamplingParams
from pdf2image import convert_from_path

# --- 1. ENGINE CONFIG ---
# Thinking models are memory-heavy; we keep parallelism (max_num_seqs) low.
llm = LLM(
    model="Qwen/Qwen3-VL-8B-Thinking",
    max_model_len=64000, 
    max_num_seqs=2, 
    trust_remote_code=True,
    mm_processor_kwargs={
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1024 * 28 * 28, # Fast but clear resolution
    }
)

sampling_params = SamplingParams(temperature=0, max_tokens=4096)

# --- 2. THINKING PROMPT ---
SYSTEM_PROMPT = """
You are a precise document extraction agent. 
First, reason step-by-step about the layout and Unique IDs. 
Then, extract the data into a JSON list inside ```json blocks.
"""

def extract_thought_and_json(text):
    """Separates internal reasoning from the structured JSON output."""
    # The JSON is usually at the end inside markdown blocks
    json_match = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    
    # Everything before the first JSON block is the 'thought' process
    thought_part = text.split("```json")[0].strip()
    
    if json_match:
        try:
            return thought_part, json.loads(json_match[-1])
        except:
            return thought_part, None
    return text, None

def process_with_thought_logging(pdf_path, chunk_size=2):
    images = convert_from_path(pdf_path, dpi=200)
    total_pages = len(images)
    master_storage = {}
    
    # Create a fresh log file for this run
    with open("extraction_thoughts.log", "w", encoding="utf-8") as log_file:
        for i in range(0, total_pages, chunk_size):
            start, end = i + 1, min(i + chunk_size, total_pages)
            print(f"Processing pages {start}-{end}...")

            chunk_images = images[i : i + chunk_size]
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
            prompt += "<|vision_start|><|vision_end|>" * len(chunk_images)
            prompt += f"Extract data from pages {start} to {end}.<|im_end|>\n<|im_start|>assistant\n"

            # Execute batch (using vLLM's internal batching)
            outputs = llm.generate([prompt], sampling_params, multi_modal_data={"image": chunk_images})
            raw_text = outputs[0].outputs[0].text

            # Split output
            thought, data = extract_thought_and_json(raw_text)

            # Log thoughts to file
            log_file.write(f"=== CHUNK: PAGES {start}-{end} ===\n")
            log_file.write(f"REASONING:\n{thought}\n")
            log_file.write(f"PARSING STATUS: {'SUCCESS' if data else 'FAILED'}\n")
            log_file.write("-" * 50 + "\n\n")
            log_file.flush() # Ensure it writes to disk immediately

            if data:
                for entry in data:
                    uid = entry.get("unique_id")
                    if uid in master_storage:
                        master_storage[uid]["description"] += " " + entry.get("description", "")
                    else:
                        master_storage[uid] = entry
            else:
                print(f"⚠️ JSON Parse error on pages {start}-{end}. Check the .log file.")

    return list(master_storage.values())

# --- 3. RUN ---
if __name__ == "__main__":
    pdf_file = "my_complex_doc.pdf"
    final_data = process_with_thought_logging(pdf_file)
    
    with open("final_output.json", "w") as f:
        json.dump(final_data, f, indent=4)
    
    print("Done. See 'extraction_thoughts.log' to debug the AI's logic.")

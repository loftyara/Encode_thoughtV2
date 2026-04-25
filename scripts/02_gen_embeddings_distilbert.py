# scripts/02_gen_embeddings_distilbert.py
import os
import torch
import gc
from transformers import AutoTokenizer, AutoModel

# ==========================================
# CONFIGURATION (TUNE THESE)
# ==========================================
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512        # Hard limit for DistilBERT positional embeddings

# Processing Settings
BATCH_SIZE = 256        # Stories per GPU forward pass (Maximize VRAM usage)
CHUNK_SIZE = 5000      # Stories saved per file (Minimize file count)

# LIMIT DATASET SIZE TO SAVE DISK SPACE
# Set to None to process ALL stories. 
# Set to an integer (e.g., 100000) to stop after N stories.
MAX_TRAIN_STORIES = 100000   # Limit Train set (e.g., 100k stories ~ 10-15GB with float16)
MAX_VAL_STORIES = 5000       # Limit Val set (None = process all ~20k stories, very small)

SEPARATOR = "<|endoftext|>"  # Story separator token

# Paths (Auto-detected relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# ==========================================
# LOGIC
# ==========================================

def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL: CUDA is not available.")
    return torch.device("cuda")

def stream_stories(filepath, max_stories=None):
    """
    Generator that yields stories one by one.
    Stops automatically if max_stories limit is reached.
    """
    print(f"Streaming stories from {filepath}...")
    current_story_lines = []
    count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            
            if stripped == SEPARATOR:
                if current_story_lines:
                    yield " ".join(current_story_lines)
                    current_story_lines = []
                    count += 1
                    if max_stories is not None and count >= max_stories:
                        return
            else:
                if stripped:
                    current_story_lines.append(stripped)
        
        # Yield the last story if file doesn't end with separator
        if current_story_lines:
            yield " ".join(current_story_lines)
            count += 1

def process_and_save_chunk(model, tokenizer, stories_chunk, chunk_idx, output_prefix, device):
    if not stories_chunk:
        return

    model.eval()
    
    # Lists to accumulate results in RAM for this chunk
    chunk_embeddings = []
    chunk_texts = []
    
    print(f"Processing chunk {chunk_idx} ({len(stories_chunk)} stories)...")
    
    # Process in batches within the chunk
    for i in range(0, len(stories_chunk), BATCH_SIZE):
        batch_stories = stories_chunk[i:i+BATCH_SIZE]
        
        # 1. Tokenize and Move to GPU
        encodings = tokenizer(
            batch_stories,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)
        
        # 2. Calculate Embeddings (GPU Usage)
        with torch.no_grad():
            outputs = model(**encodings)
            # Move to CPU immediately
            batch_embeds = outputs.last_hidden_state.cpu()
            masks = encodings.attention_mask.cpu()
        
        # 3. Extract, Convert to Float16, and Store in RAM
        for j in range(len(batch_stories)):
            seq_len = masks[j].sum().item()
            # Extract valid tokens
            story_embed = batch_embeds[j, :seq_len, :]
            # COMPRESS: Convert to float16 to save 50% disk space
            story_embed_fp16 = story_embed.to(torch.float16)
            
            chunk_embeddings.append(story_embed_fp16)
            chunk_texts.append(batch_stories[j])
        
        # 4. AGGRESSIVE CLEANUP GPU & RAM IMMEDIATELY after each batch
        del encodings, outputs, batch_embeds, masks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save accumulated chunk to disk
    save_data = {
        "embeddings": chunk_embeddings,  # Now contains float16 tensors
        "texts": chunk_texts
    }
    
    filename = f"{output_prefix}_chunk_{chunk_idx:04d}.pt"
    filepath = os.path.join(PROCESSED_DATA_DIR, filename)
    
    print(f"Saving chunk {chunk_idx} to {filename}...")
    torch.save(save_data, filepath)
    
    # Final cleanup for RAM after saving the whole chunk
    del save_data, chunk_embeddings, chunk_texts, stories_chunk
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")
    
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    
    model_short_name = MODEL_NAME.split('/')[-1]
    
    # Define datasets with their specific limits
    datasets = [
        ("train", "train.txt", MAX_TRAIN_STORIES),
        ("val", "val.txt", MAX_VAL_STORIES)
    ]
    
    for ds_name, filename, max_limit in datasets:
        input_path = os.path.join(RAW_DATA_DIR, filename)
        if not os.path.exists(input_path):
            print(f"[SKIP] {input_path} not found.")
            continue
            
        limit_str = f" (Limit: {max_limit})" if max_limit else " (All)"
        output_prefix = os.path.join(PROCESSED_DATA_DIR, f"embeddings_{model_short_name}_{ds_name}")
        
        print(f"\n--- Processing {ds_name.upper()} dataset{limit_str} ---")
        
        story_count = 0
        chunk_count = 0
        current_chunk = []
        
        # Stream stories with limit
        for story in stream_stories(input_path, max_stories=max_limit):
            current_chunk.append(story)
            story_count += 1
            
            if len(current_chunk) >= CHUNK_SIZE:
                process_and_save_chunk(model, tokenizer, current_chunk, chunk_count, output_prefix, device)
                current_chunk = []
                chunk_count += 1
        
        # Save remaining stories
        if current_chunk:
            process_and_save_chunk(model, tokenizer, current_chunk, chunk_count, output_prefix, device)
            chunk_count += 1
            
        print(f"--- Finished {ds_name.upper()}: {story_count} stories in {chunk_count} files. ---")

    print("\n--- Embedding generation completed successfully. ---")
    print(f"Files are located in: {os.path.abspath(PROCESSED_DATA_DIR)}")
    print("Note: Embeddings are saved in float16 format to save disk space.")

if __name__ == "__main__":
    main()

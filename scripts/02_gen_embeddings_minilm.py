import os
import torch
import gc
from transformers import AutoTokenizer, AutoModel

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 512        # Hard limit for positional embeddings

# Processing Settings (Same as DistilBERT run)
BATCH_SIZE = 256        # Stories per GPU forward pass
CHUNK_SIZE = 5000       # Stories saved per file

# LIMIT DATASET SIZE
MAX_TRAIN_STORIES = 100000
MAX_VAL_STORIES = 5000

SEPARATOR = "<|endoftext|>"  # Story separator token

# Paths
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
    if current_story_lines:
        yield " ".join(current_story_lines)
        count += 1

def process_and_save_chunk(model, tokenizer, stories_chunk, chunk_idx, output_prefix, device):
    if not stories_chunk:
        return
    model.eval()

    chunk_embeddings = []
    chunk_texts = []

    print(f"Processing chunk {chunk_idx} ({len(stories_chunk)} stories)...")

    for i in range(0, len(stories_chunk), BATCH_SIZE):
        batch_stories = stories_chunk[i:i+BATCH_SIZE]
        
        encodings = tokenizer(
            batch_stories,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**encodings)
            batch_embeds = outputs.last_hidden_state.cpu()
            masks = encodings.attention_mask.cpu()
        
        for j in range(len(batch_stories)):
            seq_len = masks[j].sum().item()
            story_embed = batch_embeds[j, :seq_len, :]
            story_embed_fp16 = story_embed.to(torch.float16)
            
            chunk_embeddings.append(story_embed_fp16)
            chunk_texts.append(batch_stories[j])
        
        del encodings, outputs, batch_embeds, masks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_data = {
        "embeddings": chunk_embeddings,
        "texts": chunk_texts
    }

    filename = f"{output_prefix}_chunk_{chunk_idx:04d}.pt"
    filepath = os.path.join(PROCESSED_DATA_DIR, filename)
    print(f"Saving chunk {chunk_idx} to {filename}...")
    torch.save(save_data, filepath)

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
        
        for story in stream_stories(input_path, max_stories=max_limit):
            current_chunk.append(story)
            story_count += 1
            
            if len(current_chunk) >= CHUNK_SIZE:
                process_and_save_chunk(model, tokenizer, current_chunk, chunk_count, output_prefix, device)
                current_chunk = []
                chunk_count += 1
     
        if current_chunk:
            process_and_save_chunk(model, tokenizer, current_chunk, chunk_count, output_prefix, device)
            chunk_count += 1
             
        print(f"--- Finished {ds_name.upper()}: {story_count} stories in {chunk_count} files. ---")

    print("\n--- Embedding generation completed successfully. ---")
    print(f"Files are located in: {os.path.abspath(PROCESSED_DATA_DIR)}")
    print("Note: Embeddings are saved in float16 format to save disk space.")

if __name__ == "__main__":
    main()

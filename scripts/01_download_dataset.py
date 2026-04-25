# scripts/01_download_dataset.py
import os
from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "roneneldan/TinyStories"
RAW_DATA_DIR = "data/raw"

# Exact filenames from the repository root
FILES_TO_DOWNLOAD = [
    {"repo_filename": "TinyStories-train.txt", "save_as": "train.txt"},
    {"repo_filename": "TinyStories-valid.txt", "save_as": "val.txt"}
]

def main():
    print(f"--- Starting dataset download from {REPO_ID} ---")
    
    # Create directory if it doesn't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print(f"Directory ensured: {RAW_DATA_DIR}")

    for file_info in FILES_TO_DOWNLOAD:
        repo_file = file_info["repo_filename"]
        local_filename = file_info["save_as"]
        local_path = os.path.join(RAW_DATA_DIR, local_filename)

        if os.path.exists(local_path):
            print(f"[SKIP] {local_filename} already exists at {local_path}")
            continue

        print(f"[DOWNLOAD] Fetching {repo_file}...")
        try:
            # Download file from Hugging Face Hub
            # Note: force_filename is deprecated, so we download with original name and rename manually if needed
            # But hf_hub_download with local_dir usually saves with the repo filename.
            # We will download and then rename.
            
            temp_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_file,
                repo_type="dataset",
                local_dir=RAW_DATA_DIR
            )
            
            # Rename to our standard name if different
            if os.path.basename(temp_path) != local_filename:
                os.rename(temp_path, local_path)
                print(f"[SUCCESS] Downloaded and renamed to {local_path}")
            else:
                print(f"[SUCCESS] Saved to {temp_path}")
                
        except Exception as e:
            print(f"[ERROR] Failed to download {repo_file}: {e}")
            return

    print("--- Dataset download completed successfully. ---")
    print(f"Files available in: {os.path.abspath(RAW_DATA_DIR)}")
    print("Next step: Run scripts/02_gen_embeddings_*.py")

if __name__ == "__main__":
    main()

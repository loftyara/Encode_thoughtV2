# src/dataset.py
import os
import torch
import glob
import re
from torch.utils.data import Dataset
import gc

class StoryEmbeddingDataset(Dataset):
    def __init__(self, data_dir, model_name_filter=None, split="train", max_samples=None, preload=False):
        """
        Initializes the dataset.
        
        Args:
            data_dir: Path to processed data directory.
            model_name_filter: Substring to filter files (e.g., "distilbert").
            split: "train" or "val".
            max_samples: Max number of samples to use. If None, uses all found.
            preload: If True, loads data into RAM lists immediately (Fast, uses RAM).
                     If False, builds an index map and loads from disk on demand (Slow, saves RAM).
        """
        self.data_dir = data_dir
        self.split = split
        self.model_filter = model_name_filter
        self.max_samples = max_samples
        self.preload = preload
        
        # Find matching chunk files
        pattern = f"embeddings_*_{split}_chunk_*.pt"
        if model_name_filter:
            all_files = glob.glob(os.path.join(data_dir, pattern))
            self.chunk_files = [f for f in all_files if model_name_filter in os.path.basename(f)]
        else:
            self.chunk_files = glob.glob(os.path.join(data_dir, pattern))
            
        if not self.chunk_files:
            raise FileNotFoundError(f"No chunk files found for split='{split}' and filter='{model_name_filter}'")
        
        # Sort files by index
        def extract_index(filename):
            basename = os.path.basename(filename)
            match = re.search(r'chunk_(\d+)\.pt$', basename)
            return int(match.group(1)) if match else 0
        self.chunk_files.sort(key=extract_index)
        
        print(f"Found {len(self.chunk_files)} chunk files for {split}.")
        
        if self.preload:
            # --- PRELOAD MODE: Load data into RAM lists ---
            print("Mode: PRELOAD (Loading data into RAM)...")
            self.embeddings_list = []
            self.texts_list = []
            current_count = 0
            
            for file_path in self.chunk_files:
                if self.max_samples and current_count >= self.max_samples:
                    break
                
                try:
                    data = torch.load(file_path, map_location='cpu', weights_only=False)
                    n_items = len(data['embeddings'])
                    
                    # Determine how many items to take from this file
                    remaining = (self.max_samples - current_count) if self.max_samples else n_items
                    take_count = min(n_items, remaining)
                    
                    # Extend lists
                    self.embeddings_list.extend(data['embeddings'][:take_count])
                    self.texts_list.extend(data['texts'][:take_count])
                    
                    current_count += take_count
                    del data
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            
            self.total_size = len(self.embeddings_list)
            print(f"Loaded {self.total_size} samples into RAM.")
            
        else:
            # --- LAZY MODE: Build index map only ---
            print("Mode: LAZY LOAD (Indexing files)...")
            self.index_map = [] # List of tuples: (file_path, local_idx)
            current_count = 0
            
            for file_path in self.chunk_files:
                if self.max_samples and current_count >= self.max_samples:
                    break
                
                try:
                    # Load briefly to get count
                    data = torch.load(file_path, map_location='cpu', weights_only=False)
                    n_items = len(data['embeddings'])
                    
                    remaining = (self.max_samples - current_count) if self.max_samples else n_items
                    take_count = min(n_items, remaining)
                    
                    for i in range(take_count):
                        self.index_map.append((file_path, i))
                    
                    current_count += take_count
                    del data
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            
            self.total_size = len(self.index_map)
            print(f"Indexed {self.total_size} samples.")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if self.preload:
            # Direct access from RAM
            return {
                "embeddings": self.embeddings_list[idx],
                "text": self.texts_list[idx],
                "length": self.embeddings_list[idx].shape[0]
            }
        else:
            # Load from disk
            file_path, local_idx = self.index_map[idx]
            try:
                data = torch.load(file_path, map_location='cpu', weights_only=False)
                emb = data['embeddings'][local_idx]
                txt = data['texts'][local_idx]
                del data
                return {
                    "embeddings": emb,
                    "text": txt,
                    "length": emb.shape[0]
                }
            except Exception as e:
                raise RuntimeError(f"Failed to load item {idx}: {e}")

    def clear_cache(self):
        if self.preload:
            self.embeddings_list = []
            self.texts_list = []
        gc.collect()

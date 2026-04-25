# Scripts

Contains utility programs for dataset preparation and precomputed embedding generation. These scripts handle the initial pipeline steps before model training.

## Files
- `01_download_dataset.py` - Downloads the TinyStories dataset and saves `train.txt` / `val.txt` to `../data/raw/`
- `02_gen_embeddings_bertmini.py` - Generates embeddings using `prajjwal1/bert-mini` (256d)
- `02_gen_embeddings_distilbert.py` - Generates embeddings using `distilbert-base-uncased` (768d)
- `02_gen_embeddings_jina.py` - Generates embeddings using `jina-embeddings-v2-small-en` (512d)
- `02_gen_embeddings_minilm.py` - Generates embeddings using `all-MiniLM-L6-v2` (384d)
- `02_gen_embeddings_tinybert.py` - Generates embeddings using `TinyBERT_General_4L_312D` (312d)

## Usage
1. Activate virtual environment
2. Download raw data:
```bash
python 01_download_dataset.py
```

Run embedding generators one by one:
```bash
python 02_gen_embeddings_bertmini.py
python 02_gen_embeddings_distilbert.py
python 02_gen_embeddings_minilm.py
python 02_gen_embeddings_tinybert.py
python 02_gen_embeddings_jina.py
```

Notes

Each script reads from ../data/raw/ and writes chunked .pt files to ../data/processed/
Embeddings are saved in float16 precision to reduce disk I/O and storage
Requires CUDA-capable GPU. VRAM usage varies by base model (8–16 GB recommended)
Run scripts sequentially to avoid VRAM conflicts
	

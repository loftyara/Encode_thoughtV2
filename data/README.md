# Data Directory

This directory contains all dataset-related files for the Encode_thoughtV2 project.

## Structure
```
data/
├── raw/          # Original TinyStories dataset files
└── processed/    # Precomputed token embeddings (.pt chunks)
```

## Subdirectories

- [raw/](raw/README.md) — Raw text data from TinyStories (train.txt, val.txt)
- [processed/](processed/README.md) — Precomputed embeddings for each base encoder

## Usage

1. Download raw data via `scripts/01_download_dataset.py`
2. Generate embeddings via `scripts/02_gen_embeddings_*.py`
3. Trained models consume processed embeddings from `processed/`

## Notes

- Processed embeddings are stored as float16 `.pt` chunks to reduce disk I/O
- Each chunk file contains batched sequences: shape `(B, T, d_in)` where `d_in` depends on the base encoder
- Delete individual chunks to free space; they can be regenerated from raw data

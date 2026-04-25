# Raw Dataset

This directory contains the original, unprocessed TinyStories dataset files.

## Files

- `train.txt` - Training set (~2.8M short stories). Each story ends with the `<|endoftext|>` token.
- `val.txt` - Validation set (held-out stories for evaluation). Same format as training.

## Format

Plain UTF-8 text. Stories are concatenated with the `<|endoftext|>` separator. No additional preprocessing or tokenization has been applied.

## Usage

These files are loaded by `scripts/01_download_dataset.py` and consumed by the embedding generation scripts (`scripts/02_gen_embeddings_*.py`). Do not modify manually.

## Source

TinyStories dataset: https://huggingface.co/datasets/roneneldan/TinyStories

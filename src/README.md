# Source Directory (src/)

Contains the core model implementation, dataset handling, training pipelines, and analysis scripts for the Encode_thoughtV2 project.

## Core Libraries
- `model.py` – Main `EncodeThoughtModel` architecture (Set Transformer encoder with ISAB, Learnable Queries bottleneck, Transformer decoder with positional encoding)
- `dataset.py` – Custom `StoryEmbeddingDataset` class for loading chunked `.pt` embeddings, supporting PRELOAD and LAZY modes, attention masking, and dynamic sequence slicing

## Training Scripts (`01_train_*.py`)
One script per base encoder. Each script:
- Loads precomputed embeddings from `../data/processed/`
- Initializes the model with the matching `input_dim`
- Trains with MSE loss + attention masking on active tokens
- Saves the best checkpoint to `../checkpoints/`
Files: `01_train_bertmini.py`

## Analysis Scripts (`02_analyze_slots_*.py`)
One script per trained model. Each script:
- Loads the best checkpoint and validation samples
- Reconstructs embeddings from the slot matrix
- Decodes embeddings to text via nearest-neighbor token search in the base encoder's vocabulary
- Computes cosine similarity and prints side-by-side original/recovered text comparisons
Files: `02_analyze_slots_bertmini.py`

## Typical Workflow
1. Ensure embeddings exist in `../data/processed/`
2. Run training: `python 01_train_<model>.py`
3. Run analysis: `python 02_analyze_slots_<model>.py`
4. Check `../checkpoints/best_<model>.pt` for saved weights

## Notes
- All scripts import `model.py` and `dataset.py` via relative paths
- Configuration (MODEL_FILTER, MAX_TRAIN_SAMPLES, batch size, epochs) is defined per script
- Requires PyTorch with CUDA support and internet access for initial Hugging Face encoder downloads
- Analysis scripts do not require GPU for inference if run with CPU fallback, but GPU is recommended for speed

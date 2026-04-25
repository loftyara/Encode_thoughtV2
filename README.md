# Encode Thought V2
*“Thoughts die the moment they are embodied by words.” A. Schopenhauer*

A neural architecture for extracting the invariant semantic core of text into a compact, order-invariant matrix of learnable slots. Instead of predicting the next token autoregressively, Encode_thoughtV2 compresses sequences of base encoder embeddings into a fixed semantic representation and reconstructs them via a parallel transformer decoder. The pipeline is model-agnostic and operates on top of frozen base encoders.
Full Paper: docs/Encode_thought.pdf
Version 1 (Theoretical): https://github.com/loftyara/Encode_thought/ or https://loftyara.github.io/encode_thought.html

## KEY HIGHLIGHTS
Model-Agnostic Pipeline: Compatible with any transformer encoder. Currently optimized and validated with prajjwal1/bert-mini (256d).

Ultra-Minimal Configuration: 1 slot, internal dimension 64, 1 encoder/decoder layer, 4 induced points. Total trainable parameters: approximately 0.8 to 0.9 million (less than 2 percent of the base LLM).

Hybrid Loss and Regularization: CrossEntropy (alpha=0.2) plus CosineLoss (beta=1.0) with label smoothing (0.1) and Context Dropout (p=0.15) for stable parallel training.

Phase 1 Complete: Achieves 80 to 90 percent lexical reconstruction and greater than 0.99 cosine similarity in Teacher Forcing (Corrected) mode.

Transparent Limitations: Autoregressive generation (AR and Raw AR) currently collapses into repetition loops due to Exposure Bias. Stabilizing closed-loop generation is the sole focus of next phase.

## ARCHITECTURE OVERVIEW
Text → Token Embeddings → Set Transformer Encoder → Hidden States  
                                                    ↓  
Learnable Queries → Cross-Attention → Slot Matrix (N × D)  
                                                    ↓  
Transformer Decoder → Text reconstruction  

## INSTALLATION AND SETUP
Prerequisites:
    Python 3.12 or higher
    NVIDIA GPU with CUDA support (16 GB VRAM recommended)
    Git

## Setup steps:
Clone the repository:
```
git clone https://github.com/loftyara/Encode_thoughtV2.git
```
Navigate to the project directory:
```
cd Encode_thoughtV2
```
Create virtual environment:
```
python -m venv venv
```
Activate virtual environment:
Windows:
```
venv\Scripts\activate
```
Linux/macOS:
```
source venv/bin/activate
```
Install PyTorch:
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Install dependencies:
```
pip install -r requirements.txt
```

##QUICK START
Prepare Dataset:
```
cd scripts
python 01_download_dataset.py
```
Generate Embeddings (run for each base model):
```
python 02_gen_embeddings_bertmini.py
python 02_gen_embeddings_distilbert.py
python 02_gen_embeddings_minilm.py
python 02_gen_embeddings_tinybert.py
python 02_gen_embeddings_jina.py
```
Outputs are saved as chunked .pt files in ../data/processed/
Train the Slot Model:
```
cd ../src
python 01_train_bertmini.py
```
Repeat for other models. Checkpoints are saved to ../checkpoints/
Analyze and Reconstruct:
```
python 02_analyze_slots_bertmini.py
```
Output: Side-by-side comparison of original vs. recovered text, plus cosine similarity metrics.

## PROJECT STRUCTURE
Encode_thoughtV2/  
  checkpoints/ - Saved model weights (.pt)  
  data/  
&nbsp;&nbsp;&nbsp;    raw/ - TinyStories train.txt and val.txt  
&nbsp;&nbsp;&nbsp;    processed/ - Chunked embeddings (.pt)  
  docs/ - Documentation and paper (PDF)  
  scripts/ - Dataset download and embedding generation  
  src/ - Training scripts, analysis, model/dataset libraries  
&nbsp;&nbsp;&nbsp;    model.py  
&nbsp;&nbsp;&nbsp;    dataset.py  
&nbsp;&nbsp;&nbsp;    01_train_\*.py  
&nbsp;&nbsp;&nbsp;    02_analyze_slots_\*.py  
  README.md  

## CURRENT EXPERIMENTAL RESULTS (TinyStories + bert-mini)
|Mode|Context Source|Lexical Accuracy|Status|
|----|--------------|----------------|------|
|Corrected (Teacher Forcing)|Ground Truth|80 to 90 %|Preserves plot, entities, and semantics. Minor subtoken artifacts and local repetitions at sentence boundaries|
|AR (Quantized Context)|Own Predictions|Approximately 0%|Collapses into high-frequency token loops after 5 to 10 steps|
|Raw AR|Raw Embeddings|Approximately 0%|Similar collapse with semantic drift|

Diagnosis: The architecture successfully compresses and reconstructs semantics when provided with a valid context window. AR failure is strictly due to Exposure Bias (distribution shift between training and inference), not capacity limits or architectural flaws. Scaling parameters does not resolve this; it requires a shift to sequence-level training paradigms.


## ROADMAP (NEXT PHASE: AR STABILIZATION)
All secondary directions are paused. The next iteration focuses exclusively on closing the autoregressive loop:
- Inference-Time Decoding: Test repetition_penalty, adaptive temperature, top_p/top_k, and context window truncation to break repetition attractors.
- Feedback Normalization and Anchoring: L2-normalize raw embeddings before feedback; implement periodic cross-attention resets to slots to dampen distribution drift.
- Sequence-Level Training: Transition from Teacher Forcing to closed-loop learning via Truncated BPTT, Online Scheduled Sampling, or Minimum Risk Training.

We do not know which path will work or if stable recovery is achievable with this architecture. This is an open experimental task. Scaling model size or epochs is not a priority until the basic slots-to-text loop is closed.


## CONTACT
Author: Dmitri Lyubimkov  
Email: loftlong@gmail.com  
GitHub: @loftyara  


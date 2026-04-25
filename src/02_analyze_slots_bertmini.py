import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel
from model import EncodeThoughtModel
from dataset import StoryEmbeddingDataset

CHECKPOINT_PATH = "../checkpoints/best_bert-mini.pt"
MODEL_NAME = "prajjwal1/bert-mini"
DATA_DIR = "../data/processed"
MODEL_FILTER = "bert-mini"
NUM_SAMPLES_TO_CHECK = 5
DEVICE = "cuda"
MAX_SEQ_LEN = 512

# DIAGNOSTIC: Limit decoder context to last N tokens.
# Prevents unbounded error accumulation & attention dilution.
CONTEXT_WINDOW = 64

def get_vocab_embeddings(base_model):
    emb = base_model.embeddings.word_embeddings.weight.detach().clone()
    return emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)

def find_nearest_tokens(recon_embeds, vocab_emb):
    q_norm = recon_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    q = recon_embeds / q_norm
    sims = q @ vocab_emb.T
    return torch.argmax(sims, dim=-1)

def generate_sequence_ar(model, x, max_len, base_model, vocab_emb):
    batch_size = x.size(0)
    with torch.no_grad():
        encoded = model.encoder(x)
        slots = model.bottleneck(encoded)

    bos = model.bos_token.expand(batch_size, -1, -1)
    current_context = bos
    generated_ids = []
    embedding_layer = base_model.embeddings.word_embeddings

    for _ in range(max_len):
        with torch.no_grad():
            _, rec_emb, _ = model(x, target=current_context, add_bos=False)
            next_emb_pred = rec_emb[:, -1:, :]
            
            next_id = find_nearest_tokens(next_emb_pred, vocab_emb)
            generated_ids.append(next_id)
            
            next_emb = embedding_layer(next_id)
            current_context = torch.cat([current_context, next_emb], dim=1)
            
            # CONTEXT WINDOW LIMIT
            if current_context.shape[1] > CONTEXT_WINDOW:
                current_context = current_context[:, -CONTEXT_WINDOW:, :]
                
    return torch.stack(generated_ids, dim=1)

def generate_sequence_raw(model, x, max_len, base_model, vocab_emb):
    batch_size = x.size(0)
    with torch.no_grad():
        encoded = model.encoder(x)
        slots = model.bottleneck(encoded)

    bos = model.bos_token.expand(batch_size, -1, -1)
    current_context = bos
    generated_ids = []

    for _ in range(max_len):
        with torch.no_grad():
            _, rec_emb, _ = model(x, target=current_context, add_bos=False)
            next_emb_pred = rec_emb[:, -1:, :]
            
            next_id = find_nearest_tokens(next_emb_pred, vocab_emb)
            generated_ids.append(next_id)
            
            current_context = torch.cat([current_context, next_emb_pred], dim=1)
            
            # CONTEXT WINDOW LIMIT
            if current_context.shape[1] > CONTEXT_WINDOW:
                current_context = current_context[:, -CONTEXT_WINDOW:, :]
                
    return torch.stack(generated_ids, dim=1)

def generate_sequence_corrected(model, x, max_len, base_model, tokenizer, orig_text, vocab_emb):
    device = x.device
    tokens = tokenizer.encode(orig_text, add_special_tokens=False, truncation=True, max_length=max_len)
    orig_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    length = min(orig_ids.size(1), max_len)
    target_emb = x[:, :length-1, :]

    with torch.no_grad():
        _, rec_emb, _ = model(x, target=target_emb)
        rec_slice = rec_emb[:, :length, :]
        pred_ids = find_nearest_tokens(rec_slice, vocab_emb)
        
    return pred_ids

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    print(f"Using device: {DEVICE}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    base_model = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)
    base_model.eval()
    vocab_emb = get_vocab_embeddings(base_model)

    print("Loading trained EncodeThought model...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    cfg = checkpoint.get("config", {})
    
    input_dim = cfg.get("input_dim", 256)
    dim_model = cfg.get("dim_model", 128)
    num_heads = cfg.get("num_heads", 2)
    num_encoder_layers = cfg.get("num_encoder_layers", 2)
    num_inds = cfg.get("num_inds", 4)
    num_slots = cfg.get("num_slots", 2)
    num_decoder_layers = cfg.get("num_decoder_layers", 2)
    
    dropout_slots = cfg.get("dropout_slots", 0.0)
    dropout_dec = cfg.get("dropout_dec", 0.0)
    word_dropout = cfg.get("word_dropout", 0.0)

    model = EncodeThoughtModel(
        input_dim=input_dim, dim_model=dim_model, num_heads=num_heads,
        num_encoder_layers=num_encoder_layers, num_inds=num_inds, num_slots=num_slots,
        num_decoder_layers=num_decoder_layers, max_seq_len=MAX_SEQ_LEN,
        dropout_slots=dropout_slots, dropout_dec=dropout_dec, word_dropout=word_dropout
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    if hasattr(model, "set_tied_embeddings"):
        model.set_tied_embeddings(vocab_emb)
    model.eval()
    print(f"Model loaded. Config: Slots={num_slots}, Dim={dim_model}")
    print(f"Dropouts: Slots={dropout_slots}, Dec={dropout_dec}, Word={word_dropout}")
    print(f"Context Window Limit: {CONTEXT_WINDOW}")

    print(f"Loading {NUM_SAMPLES_TO_CHECK} validation samples...")
#    val_ds = StoryEmbeddingDataset(DATA_DIR, MODEL_FILTER, "train", NUM_SAMPLES_TO_CHECK, preload=True)
    val_ds = StoryEmbeddingDataset(DATA_DIR, MODEL_FILTER, "val", NUM_SAMPLES_TO_CHECK, preload=True)

    exact_matches_ar = 0
    exact_matches_raw = 0
    exact_matches_tf = 0
    
    for i in range(len(val_ds)):
        sample = val_ds[i]
        orig_text = sample["text"]
        
        x_full = sample["embeddings"]
        length = min(sample.get("length", len(x_full)), MAX_SEQ_LEN)
        x = x_full[:length].unsqueeze(0).to(DEVICE).to(model.encoder.input_proj.weight.dtype)
        
        print(f"\n--- Sample {i+1} (Len: {length}) ---")
        print(f"Original:\n{orig_text}")
        print("-" * 70)

        with torch.no_grad():
            gen_ids_ar = generate_sequence_ar(model, x, length, base_model, vocab_emb)
            gen_ids_ar = gen_ids_ar[0].squeeze()
            recovered_text_ar = tokenizer.decode(gen_ids_ar.cpu().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            gen_ids_raw = generate_sequence_raw(model, x, length, base_model, vocab_emb)
            gen_ids_raw = gen_ids_raw[0].squeeze()
            recovered_text_raw = tokenizer.decode(gen_ids_raw.cpu().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            gen_ids_tf = generate_sequence_corrected(model, x, length, base_model, tokenizer, orig_text, vocab_emb)
            gen_ids_tf = gen_ids_tf[0].squeeze()
            recovered_text_tf = tokenizer.decode(gen_ids_tf.cpu().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

        print(f"Recovered (AR - Quantized Context):\n{recovered_text_ar}")
        print("-" * 70)
        print(f"Recovered (Raw AR - Raw Context):\n{recovered_text_raw}")
        print("-" * 70)
        print(f"Recovered (Corrected - Original Context):\n{recovered_text_tf}")
        print("-" * 70)
        
        match_ar = recovered_text_ar.strip() == orig_text.strip()
        match_raw = recovered_text_raw.strip() == orig_text.strip()
        match_tf = recovered_text_tf.strip() == orig_text.strip()
        
        if match_ar: exact_matches_ar += 1
        if match_raw: exact_matches_raw += 1
        if match_tf: exact_matches_tf += 1
            
        print(f"AR Match: {match_ar} | Raw Match: {match_raw} | Corrected Match: {match_tf}")

    print(f"\nExact Matches (AR): {exact_matches_ar}/{len(val_ds)}")
    print(f"Exact Matches (Raw): {exact_matches_raw}/{len(val_ds)}")
    print(f"Exact Matches (Corrected): {exact_matches_tf}/{len(val_ds)}")
    
    if exact_matches_tf > max(exact_matches_ar, exact_matches_raw):
        print("\nDiagnosis: Model works with perfect context. AR failure is due to error accumulation.")
    elif exact_matches_tf == 0:
        print("\nDiagnosis: Model fails even with perfect context. Slots or decoder are broken.")
    else:
        print("\nSuccess: AR performance is consistent with TF.")

if __name__ == "__main__":
    main()

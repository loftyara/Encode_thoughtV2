import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import gc
from transformers import BertTokenizer, BertModel
from model import EncodeThoughtModel
from dataset import StoryEmbeddingDataset

MODEL_FILTER = "bert-mini"
MODEL_NAME = "prajjwal1/bert-mini"
DATA_DIR = "../data/processed"
CHECKPOINT_DIR = "../checkpoints"

MAX_TRAIN_SAMPLES = 2000
MAX_VAL_SAMPLES = 100
MAX_SEQ_LEN = 512
BATCH_SIZE = 32

EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

NUM_SLOTS = 1
DIM_MODEL = 64
NUM_HEADS = 1
NUM_ENCODER_LAYERS = 1
NUM_DECODER_LAYERS = 1
NUM_INDS = 4

DROPOUT_SLOTS = 0.0
DROPOUT_DEC = 0.0
WORD_DROPOUT = 0.0

LOSS_ALPHA = 0.2
LOSS_BETA = 1.0

# AR Stability Fixes
LABEL_SMOOTHING = 0.1
CONTEXT_DROP_PROB = 0.15

DEVICE = "cuda"
torch.backends.cudnn.benchmark = True

def prepare_pinned_dataset(dataset, limit, max_len, tokenizer):
    limit = min(limit, len(dataset))
    dim = dataset[0]["embeddings"].shape[-1]
    x = torch.zeros(limit, max_len, dim, dtype=torch.float32)
    ids = torch.zeros(limit, max_len, dtype=torch.long)
    mask = torch.zeros(limit, max_len, dtype=torch.bool)
    print(f"Preparing {limit} samples with tokenization...")
    for i in range(limit):
        item = dataset[i]
        emb = item.get("embeddings")
        if emb is None: emb = item.get("embeddings")
        length = min(emb.shape[0], max_len)
        x[i, :length] = emb[:length]
        mask[i, :length] = True
        text = item.get("text", "")
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_len)
        ids[i, :len(tokens)] = torch.tensor(tokens[:length], dtype=torch.long)
    return x, ids, mask

class CosineLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, pred, target):
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        return (1.0 - torch.clamp(torch.sum(pred_norm * target_norm, dim=-1), -1.0, 1.0)).mean()

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    base_model = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)
    base_model.eval()
    bert_embeddings = base_model.embeddings.word_embeddings.weight.detach().clone()
    embedding_layer = base_model.embeddings.word_embeddings

    train_ds = StoryEmbeddingDataset(DATA_DIR, MODEL_FILTER, "train", MAX_TRAIN_SAMPLES, preload=True)
    val_ds = StoryEmbeddingDataset(DATA_DIR, MODEL_FILTER, "val", MAX_VAL_SAMPLES, preload=True)
    X_train, Ids_train, Mask_train = prepare_pinned_dataset(train_ds, MAX_TRAIN_SAMPLES, MAX_SEQ_LEN, tokenizer)
    X_val, Ids_val, Mask_val = prepare_pinned_dataset(val_ds, MAX_VAL_SAMPLES, MAX_SEQ_LEN, tokenizer)
    input_dim = X_train.shape[-1]
    print(f"Data ready. Train: {X_train.shape}, Val: {X_val.shape}")

    model = EncodeThoughtModel(
        input_dim=input_dim, dim_model=DIM_MODEL, num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS, num_inds=NUM_INDS, num_slots=NUM_SLOTS,
        num_decoder_layers=NUM_DECODER_LAYERS, max_seq_len=MAX_SEQ_LEN,
        dropout_slots=DROPOUT_SLOTS, dropout_dec=DROPOUT_DEC, word_dropout=WORD_DROPOUT
    ).to(DEVICE)
    model.set_tied_embeddings(bert_embeddings)

    # FIX: Label smoothing prevents overconfident peaks & repetition loops in AR
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=LABEL_SMOOTHING)
    cos_criterion = CosineLoss()
    best_val_loss = float('inf')
    gc.disable()

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        total_loss, n = 0.0, 0
        indices = torch.randperm(X_train.size(0))

        for i in range(0, X_train.size(0), BATCH_SIZE):
            idx = indices[i:i+BATCH_SIZE]
            x = X_train[idx].to(DEVICE, non_blocking=True)
            ids = Ids_train[idx].to(DEVICE, non_blocking=True)
            m = Mask_train[idx].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            B, L, D = x.shape
            
            target_emb = x[:, :-1, :]
            
            # FIX: Context Dropout / Slot-Forcing
            # Forces decoder to learn slot-only generation as a fallback path.
            # Prevents collapse when AR context drifts.
            if torch.rand(1).item() < CONTEXT_DROP_PROB:
                target_emb = model.bos_token.expand(B, L - 1, -1).to(DEVICE)

            target_seq = x
            target_ids = ids
            target_mask = m

            logits, rec_emb, _ = model(x, target=target_emb)
            me = target_mask.unsqueeze(-1).expand_as(rec_emb)

            loss_ce = ce_criterion(logits[target_mask], target_ids[target_mask])
            loss_cos = cos_criterion(rec_emb[me], target_seq[me])
            loss = LOSS_ALPHA * loss_ce + LOSS_BETA * loss_cos

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        train_loss = total_loss / max(n, 1)

        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for i in range(0, X_val.size(0), BATCH_SIZE):
                x_v = X_val[i:i+BATCH_SIZE].to(DEVICE, non_blocking=True)
                ids_v = Ids_val[i:i+BATCH_SIZE].to(DEVICE, non_blocking=True)
                m_v = Mask_val[i:i+BATCH_SIZE].to(DEVICE, non_blocking=True)
                logits, rec_emb, _ = model(x_v, target=x_v[:, :-1, :])
                me = m_v.unsqueeze(-1).expand_as(rec_emb)
                val_loss += (LOSS_ALPHA * ce_criterion(logits[m_v], ids_v[m_v]) + 
                             LOSS_BETA * cos_criterion(rec_emb[me], x_v[me])).item()
                val_steps += 1
        val_loss /= max(val_steps, 1)

        elapsed = time.time() - start
        print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {elapsed:.1f}s")

#        if val_loss < best_val_loss:
        if epoch >= EPOCHS - 1:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss, 'config': {
                    'num_slots': NUM_SLOTS, 'dim_model': DIM_MODEL, 'input_dim': input_dim,
                    'num_encoder_layers': NUM_ENCODER_LAYERS, 'num_decoder_layers': NUM_DECODER_LAYERS,
                    'num_inds': NUM_INDS, 'num_heads': NUM_HEADS, 'dropout_slots': DROPOUT_SLOTS,
                    'dropout_dec': DROPOUT_DEC, 'word_dropout': WORD_DROPOUT,
                    'loss_alpha': LOSS_ALPHA, 'loss_beta': LOSS_BETA,
                    'label_smoothing': LABEL_SMOOTHING, 'context_drop_prob': CONTEXT_DROP_PROB
                }
            }, os.path.join(CHECKPOINT_DIR, f"best_{MODEL_FILTER}.pt"))
        torch.cuda.empty_cache()

    gc.enable()
    print("Training complete.")

if __name__ == "__main__":
    main()

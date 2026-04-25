import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, dim_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, dropout=0.0):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln0 = nn.LayerNorm(dim_V) if ln else None
        self.ln1 = nn.LayerNorm(dim_V) if ln else None
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        scale = math.sqrt(dim_split)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / scale, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        if self.ln0: O = self.ln0(O)
        O = O + self.dropout(F.relu(self.fc_o(O)))
        if self.ln1: O = self.ln1(O)
        return O

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, dropout=0.0):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, dropout=dropout)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, dropout=dropout)
    def forward(self, x):
        H = self.mab0(self.I.repeat(x.size(0), 1, 1), x)
        return self.mab1(x, H)

class SetTransformerEncoder(nn.Module):
    def __init__(self, dim_input, dim_model, num_heads, num_inds, num_layers, dropout=0.0, max_seq_len=512):
        super(SetTransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(dim_input, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model, max_seq_len)
        self.layers = nn.ModuleList([ISAB(dim_model, dim_model, num_heads, num_inds, ln=True, dropout=dropout) for _ in range(num_layers)])
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for layer in self.layers: x = layer(x)
        return x

class SlotBottleneck(nn.Module):
    def __init__(self, dim_model, num_slots, num_heads, dropout=0.0):
        super(SlotBottleneck, self).__init__()
        self.slots_init = nn.Parameter(torch.Tensor(1, num_slots, dim_model))
        nn.init.xavier_uniform_(self.slots_init)
        self.cross_attn = MAB(dim_model, dim_model, dim_model, num_heads, ln=True, dropout=dropout)
    def forward(self, x):
        slots = self.slots_init.repeat(x.size(0), 1, 1)
        return self.cross_attn(slots, x)

class TransformerDecoder(nn.Module):
    def __init__(self, dim_model, num_heads, num_layers, output_dim, max_seq_len, dropout=0.0):
        super(TransformerDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.dim_model = dim_model
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.Tensor(1, max_seq_len, dim_model))
        nn.init.xavier_uniform_(self.pos_embedding)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                MAB(dim_model, dim_model, dim_model, num_heads, ln=True, dropout=dropout),
                MAB(dim_model, dim_model, dim_model, num_heads, ln=True, dropout=dropout)
            ]))
        self.output_proj = nn.Linear(dim_model, output_dim)

    def forward(self, slots, input_seq):
        batch_size, seq_len, _ = input_seq.shape
        if seq_len > self.max_seq_len: raise ValueError("Seq len exceeds max")
        pos_emb = self.pos_embedding[:, :seq_len, :].expand(batch_size, -1, -1)
        x = input_seq + pos_emb
        x = self.dropout(x)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).masked_fill(torch.tril(torch.ones(seq_len, seq_len, device=x.device)) == 0, float('-inf'))
        for self_attn, cross_attn in self.layers:
            Q, K, V = self_attn.fc_q(x), self_attn.fc_k(x), self_attn.fc_v(x)
            dim_split = self_attn.dim_V // self_attn.num_heads
            Q_ = torch.cat(Q.split(dim_split, 2), 0)
            K_ = torch.cat(K.split(dim_split, 2), 0)
            V_ = torch.cat(V.split(dim_split, 2), 0)
            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split), 2)
            O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
            if self_attn.ln0: O = self_attn.ln0(O)
            O = O + self_attn.dropout(F.relu(self_attn.fc_o(O)))
            if self_attn.ln1: O = self_attn.ln1(O)
            x = cross_attn(O, slots)
        return self.output_proj(x)

class EncodeThoughtModel(nn.Module):
    def __init__(self, input_dim, dim_model, num_heads, num_encoder_layers, num_inds,
                 num_slots, num_decoder_layers, max_seq_len,
                 dropout_slots=0.0, dropout_dec=0.0, word_dropout=0.0):
        super(EncodeThoughtModel, self).__init__()
        self.encoder = SetTransformerEncoder(input_dim, dim_model, num_heads, num_inds, num_encoder_layers, dropout=dropout_dec, max_seq_len=max_seq_len)
        self.bottleneck = SlotBottleneck(dim_model, num_slots, num_heads, dropout=dropout_dec)
        self.slot_dropout = nn.Dropout(dropout_slots)
        self.decoder = TransformerDecoder(dim_model, num_heads, num_decoder_layers, output_dim=dim_model, max_seq_len=max_seq_len, dropout=dropout_dec)
        self.bos_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        nn.init.xavier_uniform_(self.bos_token)
        self.word_dropout = word_dropout
        self.input_dim = input_dim
        self.dim_model = dim_model
        self.embedding_proj = nn.Linear(dim_model, input_dim)
        self.tied_embeddings = None

    def set_tied_embeddings(self, embedding_weight):
        self.tied_embeddings = embedding_weight

    def forward(self, x, target=None, add_bos=True):
        encoded = self.encoder(x)
        slots = self.slot_dropout(self.bottleneck(encoded))
        if target is not None:
            target_proj = self.encoder.input_proj(target)
            if add_bos:
                bos_proj = self.encoder.input_proj(self.bos_token)
                decoder_input = torch.cat([bos_proj.expand(target_proj.size(0), -1, -1), target_proj], dim=1)
            else:
                decoder_input = target_proj
            if self.training and self.word_dropout > 0.0:
                mask = torch.bernoulli(torch.full(decoder_input.shape[:-1], 1.0 - self.word_dropout, device=decoder_input.device)).unsqueeze(-1)
                decoder_input = decoder_input * mask / (1.0 - self.word_dropout)
            reconstructed = self.decoder(slots, decoder_input)
        else:
            reconstructed = self.decoder(slots, x)
        projected = self.embedding_proj(reconstructed)
        logits = F.linear(projected, self.tied_embeddings) if self.tied_embeddings is not None else None
        return logits, projected, slots

    def get_slots(self, x):
        return self.bottleneck(self.encoder(x))

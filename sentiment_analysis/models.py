import torch
import torch.nn as nn
import math

# -------------------------
# RNN (LSTM/GRU) classifier
# -------------------------
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, rnn_type="lstm", bidirectional=True, num_classes=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers>1 else 0.0)
        else:
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers>1 else 0.0)
        self.pool = nn.AdaptiveAvgPool1d(1)
        factor = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * factor, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len)
        mask = (x != 0).float()  # pad idx = 0
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        outputs, _ = self.rnn(emb)  # (batch, seq_len, hidden*dirs)
        # average pooling across tokens with mask
        outputs = outputs * mask.unsqueeze(-1)
        summed = outputs.sum(dim=1)  # sum over seq
        denom = mask.sum(dim=1).unsqueeze(-1).clamp(min=1.0)
        avg = summed / denom
        logits = self.fc(avg)
        return logits

# -------------------------
# Simple Transformer classifier
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, nhead=4, num_encoder_layers=2,
                 dim_feedforward=256, max_len=256, num_classes=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len)
        mask = (x == 0)  # pad tokens

        emb = self.embedding(x)
        emb = self.pos_enc(emb)

        # 🚫 No src_key_padding_mask here to avoid MPS nested tensor bug
        out = self.transformer(emb)  # (batch, seq_len, embed_dim)

        # Zero out pad positions manually
        mask_float = (~mask).unsqueeze(-1).float()
        out = out * mask_float

        # Mean pool over non-pad tokens
        summed = out.sum(dim=1)
        denom = mask_float.sum(dim=1).clamp(min=1.0)
        avg = summed / denom

        logits = self.classifier(avg)
        return logits

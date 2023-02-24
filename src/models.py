#!/usr/bin/env python
import torch
from torch import nn
from transformers import AutoModel


def _generate_mean_emb(sents):
    # get the mean embedding of other sentences except current sentence at each step
    mean_emb = list()
    mean_emb.append(torch.mean(sents[1:], 0))  # for the first sent
    for i in range(1, sents.size(0) - 1):
        mean_emb.append(torch.mean(torch.cat([sents[:i], sents[(i+1):]], 0), 0))
    mean_emb.append(torch.mean(sents[:-1], 0))  # for the final sent
    return torch.stack(mean_emb, 0)


class Encoder(nn.Module):
    def __init__(self, pre_encoder, k=4, dropout=0.1):

        super(Encoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(pre_encoder)
        hidden_size = self.encoder.config.hidden_size
        self.amplifier_ln0 = nn.Linear(hidden_size, hidden_size * 4)
        self.amplifier_ln1 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size)
        self.sal_cls = nn.Linear(hidden_size, 1, bias=False)
        self.k = k

    def amplifier(self, texts_emb):
        for _ in range(self.k):
            mean_emb = _generate_mean_emb(texts_emb)
            texts_emb = self.amplifier_ln1(self.dropout(torch.relu(self.amplifier_ln0(texts_emb - mean_emb)))) + texts_emb
        texts_emb = self.ln(texts_emb)
        return self.sal_cls(texts_emb).squeeze()

    def forward(self, reps, masks):
        texts_emb = self.encoder(input_ids=reps, attention_mask=masks).last_hidden_state
        texts_emb = texts_emb[:, 0, :]
        return texts_emb

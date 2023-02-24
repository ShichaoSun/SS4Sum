#!/usr/bin/env python
import torch


def calculate_con_loss(hidden, temperature):
    hidden_num = hidden.size(0)
    diagonal_mask = torch.eye(hidden_num).type_as(hidden).bool()
    cos_sim = torch.mm(hidden, hidden.t()) / temperature
    cos_sim = torch.masked_fill(cos_sim, diagonal_mask, float("-inf"))
    cos_sim = - torch.log_softmax(cos_sim, 1)
    target = []
    for i in range(hidden_num):
        if i % 2 == 0:
            target.append(i + 1)
        else:
            target.append(i - 1)
    target = torch.Tensor(target).type_as(hidden).long().unsqueeze(1)
    loss = torch.mean(torch.gather(cos_sim, 1, target))
    return loss

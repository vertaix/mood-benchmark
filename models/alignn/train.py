# models/alignn/train.py
import torch
import torch.nn as nn
from torch.optim import Optimizer


def train_epoch(model, dataloader, optimizer: Optimizer, device: str = 'cpu') -> float:
    model.train()
    criterion = nn.MSELoss()
    total_loss, n_batches = 0.0, 0
    for (g, lg), targets in dataloader:
        g = g.to(device)
        lg = lg.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        preds = model((g, lg)).squeeze(-1)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def eval_epoch(model, dataloader, device: str = 'cpu') -> float:
    model.eval()
    total_mae, n = 0.0, 0
    with torch.no_grad():
        for (g, lg), targets in dataloader:
            g = g.to(device)
            lg = lg.to(device)
            targets = targets.to(device)
            preds = model((g, lg)).squeeze(-1)
            total_mae += (preds - targets).abs().sum().item()
            n += targets.shape[0]
    return total_mae / max(n, 1)

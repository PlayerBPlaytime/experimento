import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

BATCH_SIZE = 4
EPOCHS = 50
LR = 3e-4
CHECKPOINT_DIR = "/kaggle/working/checkpoints"


def fft_loss(pred, target):
    pred_fft = torch.fft.fft(pred, dim=-1).abs()
    target_fft = torch.fft.fft(target, dim=-1).abs()
    return F.l1_loss(pred_fft, target_fft)


def spectral_convergence_loss(pred, target):
    return (
        torch.norm(target - pred, p="fro")
        / (torch.norm(target, p="fro") + 1e-8)
    )


def combined_loss(pred, target):
    l1 = F.l1_loss(pred, target)
    fft = fft_loss(pred, target)
    sc = spectral_convergence_loss(pred, target)
    return l1 + 0.1 * fft + 0.1 * sc


def train(model, dataset, device, progress_callback=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    if torch.cuda.device_count() > 1:
        print(f"🚀 Usando {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, EPOCHS
    )

    best_loss = float('inf')
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for lq_spec, hq_spec in dataloader:
            lq_spec = lq_spec.to(device)
            hq_spec = hq_spec.to(device)

            output = model(lq_spec)
            loss = combined_loss(output, hq_spec)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            state = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            torch.save(
                state,
                f"{CHECKPOINT_DIR}/best_model.pth"
            )

        if progress_callback:
            pct = (epoch + 1) / EPOCHS
            msg = (
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Loss: {avg_loss:.6f} | "
                f"Best: {best_loss:.6f}"
            )
            progress_callback(pct, msg)

    return losses, best_loss

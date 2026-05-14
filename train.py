import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

BATCH_SIZE     = 4
EPOCHS         = 150
LR             = 3e-4
LR_MIN         = 1e-6
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
N_FFT          = 2048
HOP_LENGTH     = 512
SAMPLE_RATE    = 44100


def log_mag_l1(pred, target):
    return F.l1_loss(
        torch.log1p(pred),
        torch.log1p(target)
    )


def spectral_convergence(pred, target):
    return (
        torch.norm(target - pred, p="fro")
        / (torch.norm(target, p="fro") + 1e-8)
    )


def high_freq_loss(pred, target, ratio=0.4):
    """
    Alta prioridad en frecuencias altas.
    Recupera el brillo y el aire de MJ.
    """
    split = int(pred.shape[2] * ratio)
    hf_pred   = pred[:, :, split:, :]
    hf_target = target[:, :, split:, :]
    return (
        F.l1_loss(hf_pred, hf_target) +
        log_mag_l1(hf_pred, hf_target)
    )


def multiscale_loss(pred, target):
    """
    Loss en múltiples resoluciones.
    Captura ataques (batería, palmas) y sustain (cuerdas, sintetizadores).
    """
    loss   = 0.0
    scales = [1.0, 0.5, 0.25, 0.125]

    for s in scales:
        if s == 1.0:
            p, t = pred, target
        else:
            size = (
                max(4, int(pred.shape[2] * s)),
                max(4, int(pred.shape[3] * s))
            )
            p = F.interpolate(pred,   size=size, mode='bilinear', align_corners=False)
            t = F.interpolate(target, size=size, mode='bilinear', align_corners=False)

        loss += F.l1_loss(p, t)
        loss += log_mag_l1(p, t)
        loss += 0.1 * spectral_convergence(p, t)

    return loss / len(scales)


def combined_loss(pred, target):
    l1 = F.l1_loss(pred, target)
    lg = log_mag_l1(pred, target)
    sc = spectral_convergence(pred, target)
    hf = high_freq_loss(pred, target)
    ms = multiscale_loss(pred, target)

    return (
        1.0 * l1 +
        0.8 * lg +
        0.2 * sc +
        0.6 * hf +
        0.5 * ms
    )


def train(model, dataset, device, progress_callback=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 2,
        pin_memory  = True,
        drop_last   = True,
    )

    if torch.cuda.device_count() > 1:
        print(f"🚀 Usando {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = LR,
        weight_decay = 1e-5,
        betas        = (0.9, 0.999)
    )

    # Warmup 10 epochs + cosine decay
    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / max(1, EPOCHS - warmup)
        return max(
            LR_MIN / LR,
            0.5 * (1.0 + np.cos(np.pi * progress))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda
    )

    best_loss = float("inf")
    losses    = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        for lq_spec, hq_spec in dataloader:
            lq_spec = lq_spec.to(device)
            hq_spec = hq_spec.to(device)

            output = model(lq_spec)

            # Asegurar mismo tamaño
            if output.shape != hq_spec.shape:
                output = F.interpolate(
                    output,
                    size  = hq_spec.shape[-2:],
                    mode  = 'bilinear',
                    align_corners=False
                )

            loss = combined_loss(output, hq_spec)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss  += loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            state = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            torch.save(state, f"{CHECKPOINT_DIR}/best_model.pth")

        if (epoch + 1) % 50 == 0:
            state = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            torch.save(state, f"{CHECKPOINT_DIR}/epoch_{epoch+1}.pth")
            print(f"\n💾 Guardado: epoch_{epoch+1}.pth")

        if progress_callback:
            pct = (epoch + 1) / EPOCHS
            msg = (
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Loss: {avg_loss:.6f} | "
                f"Best: {best_loss:.6f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )
            progress_callback(pct, msg)

    return losses, best_loss

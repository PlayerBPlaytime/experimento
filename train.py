import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BATCH_SIZE     = 8      # Máximo para 2×T4 con modelo de 50M
EPOCHS         = 150
LR             = 1e-4   # LR de fine-tune
LR_MIN         = 1e-6
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
N_FFT          = 2048
HOP_LENGTH     = 512
SAMPLE_RATE    = 44100
GRAD_CLIP      = 1.0


# ─────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────

def log_mag_l1(pred, target):
    return F.l1_loss(
        torch.log1p(pred),
        torch.log1p(target)
    )


def spectral_convergence(pred, target):
    num = torch.norm(target - pred, p="fro")
    den = torch.norm(target,        p="fro") + 1e-8
    return num / den


def high_freq_loss(pred, target, ratio=0.35):
    """
    Énfasis en frecuencias altas.
    Recupera el brillo, el aire y los armónicos
    que destruye la grabación de seminario.
    """
    split   = int(pred.shape[2] * ratio)
    p_hf    = pred[:, :, split:, :]
    t_hf    = target[:, :, split:, :]
    return F.l1_loss(p_hf, t_hf) + log_mag_l1(p_hf, t_hf)


def multiscale_loss(pred, target):
    """
    Loss en 4 resoluciones distintas.
    - Grande: estructura global (arreglo, acordes)
    - Mediana: frases musicales
    - Pequeña: ataques (batería, palmas)
    - Micro: detalle de timbre
    """
    loss   = 0.0
    scales = [1.0, 0.5, 0.25, 0.125]

    for s in scales:
        if s == 1.0:
            p, t = pred, target
        else:
            h = max(4, int(pred.shape[2] * s))
            w = max(4, int(pred.shape[3] * s))
            p = F.interpolate(pred,   size=(h, w), mode='bilinear', align_corners=False)
            t = F.interpolate(target, size=(h, w), mode='bilinear', align_corners=False)

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


# ─────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────

def train(model, dataset, device, progress_callback=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
        prefetch_factor = 2,
    )

    # Mover a device sin DataParallel
    # DataParallel tiene mucho overhead con modelos medianos
    model = model.to(device)

    # torch.compile: 2-3x más rápido en PyTorch 2.x
    # Compila el modelo a código optimizado para la T4
    try:
        print("⚡ Compilando modelo (torch.compile)...")
        model = torch.compile(model, mode="reduce-overhead")
        print("✅ Compilación exitosa. Entrenamiento acelerado.")
    except Exception as e:
        print(f"⚠️ torch.compile no disponible: {e}")
        print("   Continuando sin compilación.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = LR,
        weight_decay = 1e-5,
        betas        = (0.9, 0.999),
        eps          = 1e-8,
    )

    # Warmup 10 epochs + cosine decay
    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, EPOCHS - warmup)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(LR_MIN / LR, cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP: Automatic Mixed Precision
    # Usa float16 donde puede → 2x más rápido en T4
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float("inf")
    losses    = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        for lq_spec, hq_spec in dataloader:
            lq_spec = lq_spec.to(device, non_blocking=True)
            hq_spec = hq_spec.to(device, non_blocking=True)

            # Forward con AMP
            with torch.cuda.amp.autocast():
                output = model(lq_spec)

                if output.shape != hq_spec.shape:
                    output = F.interpolate(
                        output,
                        size  = hq_spec.shape[-2:],
                        mode  = 'bilinear',
                        align_corners=False
                    )

                loss = combined_loss(output, hq_spec)

            # Backward con AMP
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)

        # Guardar mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Guardar sin torch.compile wrapper
            raw = (
                model._orig_mod
                if hasattr(model, '_orig_mod')
                else model
            )
            torch.save(
                raw.state_dict(),
                f"{CHECKPOINT_DIR}/best_model.pth"
            )

        # Checkpoint cada 50 epochs
        if (epoch + 1) % 50 == 0:
            raw = (
                model._orig_mod
                if hasattr(model, '_orig_mod')
                else model
            )
            torch.save(
                raw.state_dict(),
                f"{CHECKPOINT_DIR}/epoch_{epoch+1}.pth"
            )
            print(f"\n💾 epoch_{epoch+1}.pth guardado")

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

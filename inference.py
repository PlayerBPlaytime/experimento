import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
import tempfile

SAMPLE_RATE     = 44100
N_FFT           = 2048
HOP_LENGTH      = 512
CHUNK_SECONDS   = 15
OVERLAP_SECONDS = 2


def load_audio(path, target_sr=SAMPLE_RATE):
    try:
        wav, sr = librosa.load(path, sr=target_sr, mono=True)
        return torch.from_numpy(wav).float(), sr
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar: {e}")


def enhance_audio(model, audio_path, device, progress_callback=None):
    # Si el modelo fue compilado, usar el original para inferencia
    raw_model = (
        model._orig_mod
        if hasattr(model, '_orig_mod')
        else model
    )
    raw_model.eval()

    wav, _ = load_audio(audio_path)

    chunk_size = SAMPLE_RATE * CHUNK_SECONDS
    overlap    = SAMPLE_RATE * OVERLAP_SECONDS
    step       = chunk_size - overlap
    window     = torch.hann_window(N_FFT)

    starts     = list(range(0, wav.shape[-1], step))
    num_chunks = len(starts)
    chunks     = []

    with torch.no_grad():
        for i, start in enumerate(starts):
            end   = min(start + chunk_size, wav.shape[-1])
            chunk = wav[start:end]

            if chunk.shape[-1] < N_FFT:
                chunk = F.pad(chunk, (0, N_FFT - chunk.shape[-1]))

            spec = torch.stft(
                chunk,
                n_fft          = N_FFT,
                hop_length     = HOP_LENGTH,
                window         = window,
                return_complex = True,
            )

            mag   = spec.abs().unsqueeze(0).unsqueeze(0).to(device)
            phase = spec.angle()

            with torch.cuda.amp.autocast():
                cleaned = raw_model(mag)

            cleaned = cleaned.squeeze(0).squeeze(0).cpu().float()

            if cleaned.shape != phase.shape:
                cleaned = F.interpolate(
                    cleaned.unsqueeze(0).unsqueeze(0),
                    size  = phase.shape,
                    mode  = 'bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)

            out_spec = cleaned * torch.exp(1j * phase)

            out_wav = torch.istft(
                out_spec,
                n_fft      = N_FFT,
                hop_length = HOP_LENGTH,
                window     = window,
                length     = chunk.shape[-1],
            )

            chunks.append((start, out_wav))

            if progress_callback:
                progress_callback(
                    (i + 1) / num_chunks,
                    f"Chunk {i+1}/{num_chunks}"
                )

    # Overlap-add
    total  = wav.shape[-1]
    final   = torch.zeros(total)
    weights = torch.zeros(total)

    for start, chunk in chunks:
        end    = min(start + chunk.shape[-1], total)
        actual = end - start
        final[start:end]   += chunk[:actual]
        weights[start:end] += 1.0

    weights = weights.clamp(min=1.0)
    final   = final / weights
    peak    = final.abs().max()
    if peak > 0:
        final = final / peak * 0.95

    out_path = tempfile.mktemp(suffix=".wav")
    sf.write(out_path, final.numpy(), SAMPLE_RATE, subtype="PCM_24")

    return out_path, SAMPLE_RATE

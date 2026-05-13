import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

SAMPLE_RATE = 48000
N_FFT = 2048
HOP_LENGTH = 512
CHUNK_SECONDS = 10
OVERLAP_SECONDS = 1


def load_audio(path, target_sr=SAMPLE_RATE):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    return wav, sr


def enhance_audio(model, audio_path, device, progress_callback=None):
    model.eval()

    wav, original_sr = load_audio(audio_path)

    chunk_size = SAMPLE_RATE * CHUNK_SECONDS
    overlap = SAMPLE_RATE * OVERLAP_SECONDS
    step = chunk_size - overlap

    chunks = []
    starts = list(range(0, len(wav), step))
    num_chunks = len(starts)

    with torch.no_grad():
        for i, start in enumerate(starts):
            end = min(start + chunk_size, len(wav))
            chunk = wav[start:end]

            if len(chunk) < N_FFT:
                chunk = F.pad(chunk, (0, N_FFT - len(chunk)))

            window = torch.hann_window(N_FFT)
            spec = torch.stft(
                chunk,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                window=window,
                return_complex=True
            )

            magnitude = spec.abs().unsqueeze(0).unsqueeze(0).to(device)
            phase = spec.angle()

            cleaned_mag = model(magnitude)
            cleaned_mag = cleaned_mag.squeeze().cpu()

            cleaned_spec = cleaned_mag * torch.exp(1j * phase)

            cleaned_wav = torch.istft(
                cleaned_spec,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                window=window,
                length=len(chunk)
            )

            chunks.append((start, cleaned_wav))

            if progress_callback:
                pct = (i + 1) / num_chunks
                progress_callback(
                    pct, f"Chunk {i+1}/{num_chunks}"
                )

    total_len = len(wav)
    final = torch.zeros(total_len)
    weights = torch.zeros(total_len)

    for start, chunk in chunks:
        end = min(start + len(chunk), total_len)
        actual = end - start
        final[start:end] += chunk[:actual]
        weights[start:end] += 1.0

    weights = weights.clamp(min=1.0)
    final = final / weights
    final = final / (final.abs().max() + 1e-8) * 0.95

    if original_sr != SAMPLE_RATE:
        final = T.Resample(
            SAMPLE_RATE, original_sr
        )(final.unsqueeze(0)).squeeze(0)

    return final, original_sr

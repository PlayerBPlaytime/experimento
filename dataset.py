import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

SAMPLE_RATE = 44100
N_FFT       = 2048
HOP_LENGTH  = 512


class MusicDataset(Dataset):
    def __init__(
        self,
        lq_dir,
        hq_dir,
        sr              = SAMPLE_RATE,
        segment_seconds = 8,
    ):
        self.sr          = sr
        self.segment_len = sr * segment_seconds
        self.pairs       = []

        lq_files = sorted(Path(lq_dir).glob("*.wav"))
        hq_files = sorted(Path(hq_dir).glob("*.wav"))

        for lq, hq in zip(lq_files, hq_files):
            self.pairs.append((str(lq), str(hq)))

        if len(self.pairs) == 0:
            raise ValueError(
                f"No se encontraron pares en {lq_dir} / {hq_dir}"
            )

        print(f"📂 Dataset: {len(self.pairs)} pares")

    def load_audio(self, path):
        wav, sr = torchaudio.load(path)

        if sr != self.sr:
            wav = T.Resample(sr, self.sr)(wav)

        # Forzar mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        return wav.squeeze(0)

    def to_spec(self, wav):
        window = torch.hann_window(N_FFT)
        spec   = torch.stft(
            wav,
            n_fft          = N_FFT,
            hop_length     = HOP_LENGTH,
            window         = window,
            return_complex = True,
        )
        return spec.abs().unsqueeze(0)

    def random_crop(self, lq, hq):
        min_len = min(lq.shape[-1], hq.shape[-1])

        if min_len > self.segment_len:
            start = np.random.randint(0, min_len - self.segment_len)
            lq    = lq[start:start + self.segment_len]
            hq    = hq[start:start + self.segment_len]
        else:
            pad = self.segment_len - min_len
            lq  = F.pad(lq, (0, pad))
            hq  = F.pad(hq, (0, pad))

        return lq, hq

    def augment(self, lq, hq):
        """
        Augmentaciones para 21 pares.
        Más variedad → mejor generalización.
        """
        # Cambio de volumen aleatorio
        gain = np.random.uniform(0.7, 1.0)
        lq   = lq * gain
        hq   = hq * gain

        # Flip temporal (50%)
        if np.random.random() < 0.5:
            lq = torch.flip(lq, dims=[-1])
            hq = torch.flip(hq, dims=[-1])

        return lq, hq

    def normalize(self, wav):
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak * 0.95
        return wav

    def __len__(self):
        # 21 × 20 = 420 muestras por epoch
        return len(self.pairs) * 20

    def __getitem__(self, idx):
        lq_path, hq_path = self.pairs[idx % len(self.pairs)]

        lq = self.load_audio(lq_path)
        hq = self.load_audio(hq_path)

        lq, hq = self.random_crop(lq, hq)
        lq, hq = self.augment(lq, hq)

        lq = self.normalize(lq)
        hq = self.normalize(hq)

        return self.to_spec(lq), self.to_spec(hq)


SeminarDataset = MusicDataset

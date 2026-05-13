import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

SAMPLE_RATE = 48000
N_FFT = 2048
HOP_LENGTH = 512


class SeminarDataset(Dataset):
    def __init__(
        self,
        lq_dir,
        hq_dir,
        sr=SAMPLE_RATE,
        segment_seconds=3,
        augment=True
    ):
        self.sr = sr
        self.segment_len = sr * segment_seconds
        self.augment = augment
        self.pairs = []

        lq_files = sorted(Path(lq_dir).glob("*.wav"))
        hq_files = sorted(Path(hq_dir).glob("*.wav"))

        for lq, hq in zip(lq_files, hq_files):
            self.pairs.append((str(lq), str(hq)))

        if len(self.pairs) == 0:
            raise ValueError(
                f"No se encontraron pares en {lq_dir} / {hq_dir}"
            )

        print(f"📂 Dataset: {len(self.pairs)} pares encontrados")

    def load_audio(self, path):
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != self.sr:
            wav = T.Resample(sr, self.sr)(wav)
        return wav.squeeze(0)

    def to_spec(self, wav):
        window = torch.hann_window(N_FFT)
        spec = torch.stft(
            wav,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=window,
            return_complex=True
        )
        return spec.abs()

    def random_crop(self, lq, hq):
        min_len = min(len(lq), len(hq))
        if min_len > self.segment_len:
            start = np.random.randint(0, min_len - self.segment_len)
            lq = lq[start:start + self.segment_len]
            hq = hq[start:start + self.segment_len]
        else:
            pad = self.segment_len - min_len
            lq = F.pad(lq, (0, pad))
            hq = F.pad(hq, (0, pad))
        return lq, hq

    def augment_audio(self, wav):
        gain = np.random.uniform(0.7, 1.0)
        wav = wav * gain
        if np.random.random() < 0.5:
            wav = -wav
        return wav

    def __len__(self):
        return len(self.pairs) * 5

    def __getitem__(self, idx):
        lq_path, hq_path = self.pairs[idx % len(self.pairs)]

        lq = self.load_audio(lq_path)
        hq = self.load_audio(hq_path)

        lq, hq = self.random_crop(lq, hq)

        if self.augment:
            hq = self.augment_audio(hq)

        lq = lq / (lq.abs().max() + 1e-8)
        hq = hq / (hq.abs().max() + 1e-8)

        lq_spec = self.to_spec(lq).unsqueeze(0)
        hq_spec = self.to_spec(hq).unsqueeze(0)

        return lq_spec, hq_spec

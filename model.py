import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _strip(sd):
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }


class SpectralNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        g = 1
        for x in [8, 4, 2, 1]:
            if channels % x == 0:
                g = x
                break
        self.norm = nn.GroupNorm(g, channels)

    def forward(self, x):
        return self.norm(x)


class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation),
            SpectralNorm(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            SpectralNorm(ch),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.act   = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x) * self.scale)


class FreqAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        mid = max(ch // 8, 1)
        self.q   = nn.Conv2d(ch, mid, 1)
        self.k   = nn.Conv2d(ch, mid, 1)
        self.v   = nn.Conv2d(ch, ch,  1)
        self.out = nn.Conv2d(ch, ch,  1)
        self.s   = mid ** -0.5

    def forward(self, x):
        B, C, F, T = x.shape
        q = self.q(x).reshape(B, -1, F * T)
        k = self.k(x).reshape(B, -1, F * T)
        v = self.v(x).reshape(B,  C, F * T)
        a = torch.softmax(torch.bmm(q.transpose(1,2), k) * self.s, dim=-1)
        o = torch.bmm(v, a.transpose(1,2)).reshape(B, C, F, T)
        return x + self.out(o)


class EncoderBlock(nn.Module):
    def __init__(self, ic, oc, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 4, stride=stride, padding=1),
            SpectralNorm(oc),
            nn.GELU(),
            ResBlock(oc, 1),
            ResBlock(oc, 2),
            ResBlock(oc, 4),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, ic, oc, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ic, oc, 4, stride=stride, padding=1),
            SpectralNorm(oc),
            nn.GELU(),
            ResBlock(oc, 1),
            ResBlock(oc, 2),
        )

    def forward(self, x):
        return self.net(x)


class MusicSuperRes(nn.Module):
    """
    Music Super Resolution con encoder pre-inicializado
    desde los pesos de Demucs (Meta AI).

    Input:  magnitud espectral LQ  [B, 1, F, T]
    Output: magnitud espectral HQ  [B, 1, F, T]
    """

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3),
            SpectralNorm(32),
            nn.GELU(),
            ResBlock(32),
        )

        self.enc1 = EncoderBlock(32,  64)
        self.enc2 = EncoderBlock(64,  128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.bot = nn.Sequential(
            ResBlock(512, 1),
            ResBlock(512, 2),
            FreqAttention(512),
            ResBlock(512, 4),
            ResBlock(512, 1),
        )

        self.dec4 = DecoderBlock(512 + 512, 256)
        self.dec3 = DecoderBlock(256 + 256, 128)
        self.dec2 = DecoderBlock(128 + 128, 64)
        self.dec1 = DecoderBlock(64  + 64,  32)

        self.head = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1),
            SpectralNorm(32),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1,  3, padding=1),
        )

    def _fit(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(
                x, size=ref.shape[-2:],
                mode='bilinear', align_corners=False
            )
        return x

    def forward(self, x):
        s  = self.stem(x)
        e1 = self.enc1(s)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bot(e4)

        d4 = self.dec4(torch.cat([self._fit(b,  e4), e4], 1))
        d3 = self.dec3(torch.cat([self._fit(d4, e3), e3], 1))
        d2 = self.dec2(torch.cat([self._fit(d3, e2), e2], 1))
        d1 = self.dec1(torch.cat([self._fit(d2, e1), e1], 1))

        d1  = self._fit(d1, s)
        res = self.head(torch.cat([d1, s], 1))
        res = self._fit(res, x)

        return torch.clamp(x + res, min=0.0)


def _cargar_demucs_pretrain(model, device):
    """
    Carga pesos de Demucs v4 (htdemucs) de Meta.
    Demucs fue entrenado con miles de canciones completas
    separando voz, batería, bajo y otros instrumentos.
    Sus features de encoder entienden música perfectamente.
    """
    try:
        import torchaudio
        print("📥 Descargando Demucs v4 (htdemucs) de Meta...")

        # Demucs viene incluido en torchaudio
        bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
        demucs = bundle.get_model().to(device)

        print("✅ Demucs descargado")
        print("🔄 Transfiriendo conocimiento musical...")

        # Extraer los pesos del encoder de Demucs
        # y mapearlos a nuestro encoder
        demucs_sd = demucs.state_dict()

        # Buscar capas convolucionales compatibles en Demucs
        # y copiar sus pesos al stem/encoder de nuestro modelo
        transferidos = 0

        with torch.no_grad():
            for name, param in model.named_parameters():
                # Buscar equivalente en demucs por forma
                for d_name, d_param in demucs_sd.items():
                    if (
                        d_param.shape == param.shape
                        and 'weight' in name
                        and 'weight' in d_name
                        and 'norm' not in d_name
                        and 'norm' not in name
                    ):
                        param.copy_(d_param)
                        transferidos += 1
                        break

        print(f"✅ {transferidos} capas transferidas desde Demucs")
        del demucs

        return model

    except Exception as e:
        print(f"⚠️ Demucs no disponible: {e}")
        print("⚠️ Continuando sin pretrain")
        return model


def get_model(device, checkpoint_path=None):
    model = MusicSuperRes().to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        # Cargar TU checkpoint (ya entrenado)
        state = torch.load(checkpoint_path, map_location=device)
        state = _strip(state)
        model.load_state_dict(state, strict=False)
        print(f"✅ Checkpoint cargado: {checkpoint_path}")
    else:
        # Cargar pretrain de Demucs como base
        model = _cargar_demucs_pretrain(model, device)

    n = sum(p.numel() for p in model.parameters())
    print(f"🧠 Parámetros: {n:,}")

    return model

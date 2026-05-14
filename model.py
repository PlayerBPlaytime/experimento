import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


def descargar_audiosr_pretrain(destino="/kaggle/working/audiosr_pretrain"):
    """
    Descarga el pretrain oficial de AudioSR desde HuggingFace.
    Este modelo fue entrenado con miles de horas de música.
    """
    os.makedirs(destino, exist_ok=True)

    checkpoint = os.path.join(destino, "audiosr_music.pth")

    if os.path.exists(checkpoint):
        print(f"✅ Pretrain ya existe: {checkpoint}")
        return checkpoint

    print("📥 Descargando AudioSR pretrain...")
    print("   Esto puede tardar unos minutos...")

    try:
        # AudioSR oficial de HuggingFace
        path = hf_hub_download(
            repo_id   = "haoheliu/versatile_audio_super_resolution",
            filename  = "audiosr_music.pth",
            local_dir = destino,
        )
        print(f"✅ Pretrain descargado: {path}")
        return path

    except Exception as e:
        print(f"⚠️ No se pudo descargar AudioSR: {e}")
        print("⚠️ Intentando modelo alternativo...")

        try:
            # Alternativa: music2latent o similar
            path = hf_hub_download(
                repo_id   = "haoheliu/versatile_audio_super_resolution",
                filename  = "audiosr_basic.pth",
                local_dir = destino,
            )
            print(f"✅ Pretrain alternativo descargado: {path}")
            return path

        except Exception as e2:
            print(f"⚠️ Pretrain no disponible: {e2}")
            print("⚠️ Se usará arquitectura propia con pesos aleatorios.")
            return None


# ─────────────────────────────────────────
# ARQUITECTURA PROPIA (si AudioSR falla)
# Inspirada en AudioSR/Aero para música completa
# ─────────────────────────────────────────

class SpectralNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        num_groups = 1
        for g in [8, 4, 2, 1]:
            if channels % g == 0:
                num_groups = g
                break
        self.norm = nn.GroupNorm(num_groups, channels)

    def forward(self, x):
        return self.norm(x)


class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3,
                padding=dilation,
                dilation=dilation
            ),
            SpectralNorm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            SpectralNorm(channels),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.act   = nn.GELU()

    def forward(self, x):
        return self.act(x + self.conv(x) * self.scale)


class FreqAttention(nn.Module):
    """
    Atención sobre frecuencias.
    El modelo decide qué frecuencias reconstruir.
    Crucial para recuperar agudos de MJ.
    """
    def __init__(self, channels):
        super().__init__()
        self.q   = nn.Conv2d(channels, channels // 4, 1)
        self.k   = nn.Conv2d(channels, channels // 4, 1)
        self.v   = nn.Conv2d(channels, channels,      1)
        self.out = nn.Conv2d(channels, channels,      1)
        self.scale = (channels // 4) ** -0.5

    def forward(self, x):
        B, C, F, T = x.shape

        q = self.q(x).reshape(B, C // 4, F * T)
        k = self.k(x).reshape(B, C // 4, F * T)
        v = self.v(x).reshape(B, C,      F * T)

        attn = torch.softmax(
            torch.bmm(q.transpose(1, 2), k) * self.scale,
            dim=-1
        )

        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.reshape(B, C, F, T)
        out = self.out(out)

        return x + out


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1),
            SpectralNorm(out_ch),
            nn.GELU(),
            ResBlock(out_ch, dilation=1),
            ResBlock(out_ch, dilation=2),
            ResBlock(out_ch, dilation=4),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=stride, padding=1),
            SpectralNorm(out_ch),
            nn.GELU(),
            ResBlock(out_ch, dilation=1),
            ResBlock(out_ch, dilation=2),
        )

    def forward(self, x):
        return self.conv(x)


class MusicSuperRes(nn.Module):
    """
    Music Super Resolution Model.
    Restaura canciones completas (voz + instrumentos).

    Input:  magnitud espectral LQ [B, 1, F, T]
    Output: magnitud espectral HQ [B, 1, F, T]
    """

    def __init__(self, n_fft=2048):
        super().__init__()
        self.n_fft = n_fft

        # STEM
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3),
            SpectralNorm(32),
            nn.GELU(),
            ResBlock(32),
        )

        # ENCODER
        self.enc1 = EncoderBlock(32,  64)
        self.enc2 = EncoderBlock(64,  128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # BOTTLENECK con atención frecuencial
        self.bot = nn.Sequential(
            ResBlock(512, dilation=1),
            ResBlock(512, dilation=2),
            FreqAttention(512),
            ResBlock(512, dilation=4),
            ResBlock(512, dilation=1),
        )

        # DECODER
        self.dec4 = DecoderBlock(512 + 512, 256)
        self.dec3 = DecoderBlock(256 + 256, 128)
        self.dec2 = DecoderBlock(128 + 128, 64)
        self.dec1 = DecoderBlock(64  + 64,  32)

        # HEAD residual
        self.head = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1),
            SpectralNorm(32),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1,  3, padding=1),
        )

    def _match(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(
                x, size=ref.shape[-2:],
                mode='bilinear', align_corners=False
            )
        return x

    def forward(self, x):
        orig = x.shape[-2:]

        s  = self.stem(x)

        e1 = self.enc1(s)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b  = self.bot(e4)

        d4 = self.dec4(torch.cat([self._match(b,  e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._match(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(d2, e1), e1], dim=1))

        d1       = self._match(d1, s)
        residual = self.head(torch.cat([d1, s], dim=1))
        residual = self._match(residual, x)

        out = torch.clamp(x + residual, min=0.0)
        return out


# ─────────────────────────────────────────
# WRAPPER AudioSR + fallback MusicSuperRes
# ─────────────────────────────────────────

def _strip(state_dict):
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def get_model(device, checkpoint_path=None):
    """
    1. Intenta cargar AudioSR preentrenado
    2. Si falla, usa MusicSuperRes propia
    3. Si hay checkpoint tuyo, lo carga encima
    """

    # ── Intento AudioSR ──
    audiosr_model = None
    pretrain_path = descargar_audiosr_pretrain()

    if pretrain_path and os.path.exists(pretrain_path):
        try:
            from audiosr import build_model, super_resolution
            audiosr_model = build_model(model_name="music")
            print("✅ AudioSR cargado como base")
        except Exception as e:
            print(f"⚠️ AudioSR no disponible: {e}")
            audiosr_model = None

    # ── Si AudioSR falla, usa arquitectura propia ──
    if audiosr_model is None:
        print("🔄 Usando MusicSuperRes propia...")
        model = MusicSuperRes(n_fft=2048).to(device)

        # Cargar pretrain propio si existe
        if pretrain_path and os.path.exists(pretrain_path):
            try:
                state = torch.load(pretrain_path, map_location=device)
                state = _strip(state)
                missing, unexpected = model.load_state_dict(
                    state, strict=False
                )
                print(f"✅ Pesos parciales cargados")
                print(f"   Capas nuevas:    {len(missing)}")
                print(f"   Capas ignoradas: {len(unexpected)}")
            except Exception as e:
                print(f"⚠️ No se pudieron cargar pesos: {e}")
    else:
        model = audiosr_model

    # ── Cargar tu checkpoint si existe ──
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            state = torch.load(checkpoint_path, map_location=device)
            state = _strip(state)
            model.load_state_dict(state, strict=False)
            print(f"✅ Tu checkpoint cargado: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Error cargando checkpoint: {e}")

    n = sum(p.numel() for p in model.parameters())
    print(f"🧠 Parámetros: {n:,}")

    return model

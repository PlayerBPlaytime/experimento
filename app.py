import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.conv(x) * 0.1


class SeminarGhostKiller(nn.Module):
    """
    U-Net para limpieza de audio de seminarios.
    Aprende a mapear espectrogramas sucios a limpios.
    """

    def __init__(self):
        super().__init__()

        # ENCODER
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ResidualBlock(64),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ResidualBlock(256),
        )

        # BOTTLENECK
        self.bottleneck = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

        # DECODER
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ResidualBlock(64),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        # OUTPUT
        self.output = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        orig_size = x.shape[-2:]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder con skip connections
        d4 = self.dec4(torch.cat([b, e4], dim=1))
        if d4.shape[-2:] != e3.shape[-2:]:
            d4 = F.interpolate(
                d4, size=e3.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        if d3.shape[-2:] != e2.shape[-2:]:
            d3 = F.interpolate(
                d3, size=e2.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        if d2.shape[-2:] != e1.shape[-2:]:
            d2 = F.interpolate(
                d2, size=e1.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        out = self.output(torch.cat([d2, e1], dim=1))
        if out.shape[-2:] != orig_size:
            out = F.interpolate(
                out, size=orig_size,
                mode='bilinear',
                align_corners=False
            )

        # Máscara multiplicativa
        return x * out + (1 - out) * x.mean()


def get_model(device, checkpoint_path=None):
    """
    Carga el modelo con o sin checkpoint.
    """
    model = SeminarGhostKiller().to(device)

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"✅ Checkpoint cargado: {checkpoint_path}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Parámetros: {n_params:,}")

    return model

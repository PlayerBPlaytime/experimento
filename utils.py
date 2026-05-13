import os
import shutil
import zipfile
import requests
from pathlib import Path


def descargar_desde_hf(url, local_path="/kaggle/working/hf_download", token=None):
    """
    Descarga un archivo desde Hugging Face usando el link directo completo.
    """

    os.makedirs(local_path, exist_ok=True)

    filename = url.split("/")[-1].split("?")[0]
    dest = os.path.join(local_path, filename)

    print(f"⬇️ Descargando desde: {url}")
    print(f"   Guardando en: {dest}")

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0))
        total_mb = total / 1e6

        print(f"   Tamaño total: {total_mb:.1f} MB")

        descargado = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                descargado += len(chunk)
                pct = (descargado / total * 100) if total else 0
                print(
                    f"\r   Progreso: {pct:.1f}% "
                    f"({descargado/1e6:.1f}/{total_mb:.1f} MB)",
                    end=""
                )

    print(f"\n✅ Descarga completa: {dest}")
    return dest


def extraer_zip(zip_path, destino="/kaggle/working/dataset"):
    if os.path.exists(destino):
        shutil.rmtree(destino)
    os.makedirs(destino)

    print(f"📦 Extrayendo ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(destino)

    print(f"✅ ZIP extraído en: {destino}")

    lq_dir = None
    hq_dir = None

    for root, dirs, files in os.walk(destino):
        for d in dirs:
            if d.lower() == 'lq':
                lq_dir = os.path.join(root, d)
            elif d.lower() == 'hq':
                hq_dir = os.path.join(root, d)

    if lq_dir is None or hq_dir is None:
        print("⚠️ No se encontraron carpetas lq/hq, buscando por nombre...")
        lq_dir = os.path.join(destino, "lq")
        hq_dir = os.path.join(destino, "hq")
        os.makedirs(lq_dir, exist_ok=True)
        os.makedirs(hq_dir, exist_ok=True)

        for f in Path(destino).rglob("*.wav"):
            name = f.stem.lower()
            if "lq" in name:
                shutil.copy(f, lq_dir)
            elif "hq" in name:
                shutil.copy(f, hq_dir)

    print(f"📂 LQ: {lq_dir}")
    print(f"📂 HQ: {hq_dir}")

    return lq_dir, hq_dir


def verificar_dataset(lq_dir, hq_dir):
    lq_files = sorted(Path(lq_dir).glob("*.wav"))
    hq_files = sorted(Path(hq_dir).glob("*.wav"))

    errores = []

    if len(lq_files) == 0:
        errores.append("❌ No hay archivos WAV en lq/")
    if len(hq_files) == 0:
        errores.append("❌ No hay archivos WAV en hq/")
    if len(lq_files) != len(hq_files):
        errores.append(
            f"❌ Diferente cantidad: lq={len(lq_files)}, hq={len(hq_files)}"
        )

    for l, h in zip(lq_files, hq_files):
        if l.name != h.name:
            errores.append(f"❌ No coinciden: {l.name} ↔ {h.name}")

    return errores, len(lq_files)


def crear_lq_sintetico(hq_dir, lq_dir):
    try:
        from pedalboard import (
            Pedalboard, Reverb,
            LowpassFilter, HighpassFilter,
            Compressor
        )
        from pedalboard.io import AudioFile
        import numpy as np
    except ImportError:
        return "❌ Instala pedalboard: pip install pedalboard"

    os.makedirs(lq_dir, exist_ok=True)

    board = Pedalboard([
        Reverb(
            room_size=0.75,
            damping=0.4,
            wet_level=0.45,
            dry_level=0.55
        ),
        HighpassFilter(cutoff_frequency_hz=180),
        LowpassFilter(cutoff_frequency_hz=5500),
        Compressor(threshold_db=-18, ratio=5),
    ])

    archivos = list(Path(hq_dir).glob("*.wav"))

    if len(archivos) == 0:
        return "❌ No hay archivos WAV en la carpeta HQ"

    for archivo in archivos:
        with AudioFile(str(archivo)) as f:
            audio = f.read(f.frames)
            sr = f.samplerate

        lq_audio = board(audio, sr)

        import numpy as np
        ruido = np.random.randn(*lq_audio.shape) * 0.008
        lq_audio = lq_audio + ruido
        lq_audio = lq_audio * 0.7
        lq_audio = lq_audio / np.abs(lq_audio).max() * 0.85

        lq_path = os.path.join(lq_dir, archivo.name)
        with AudioFile(lq_path, 'w', sr, lq_audio.shape[0]) as f:
            f.write(lq_audio)

        print(f"✅ {archivo.name}")

    return f"✅ {len(archivos)} archivos LQ creados"

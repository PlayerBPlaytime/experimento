import os
import shutil
import zipfile
from pathlib import Path


def descargar_desde_hf(repo_id, filename, local_path, token=None):
    """
    Descarga un archivo desde Hugging Face Dataset.
    
    repo_id: "tu_usuario/dataset-seminario"
    filename: "dataset.zip"
    local_path: donde guardarlo
    token: tu HF token si el dataset es privado
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        os.system("pip install huggingface_hub -q")
        from huggingface_hub import hf_hub_download

    print(f"⬇️ Descargando {filename} desde HuggingFace...")
    print(f"   Repo: {repo_id}")
    print(f"   Esto puede tardar unos minutos dependiendo del tamaño...")

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_path,
        token=token,
    )

    print(f"✅ Descargado en: {path}")
    return path


def extraer_zip(zip_path, destino="/kaggle/working/dataset"):
    """
    Extrae el ZIP y encuentra las carpetas lq/hq.
    """

    if os.path.exists(destino):
        shutil.rmtree(destino)
    os.makedirs(destino)

    print(f"📦 Extrayendo ZIP...")
    print(f"   Origen: {zip_path}")
    print(f"   Destino: {destino}")

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(destino)

    print(f"✅ ZIP extraído")

    # Buscar carpetas lq y hq
    lq_dir = None
    hq_dir = None

    for root, dirs, files in os.walk(destino):
        for d in dirs:
            if d.lower() == 'lq':
                lq_dir = os.path.join(root, d)
            elif d.lower() == 'hq':
                hq_dir = os.path.join(root, d)

    # Si no hay carpetas nombradas buscar por nombre de archivo
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
    """
    Verifica que el dataset esté bien formado.
    """

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
    """
    Crea versiones LQ sintéticas a partir de audios HQ.
    """

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

        ruido = np.random.randn(*lq_audio.shape) * 0.008
        lq_audio = lq_audio + ruido
        lq_audio = lq_audio * 0.7
        lq_audio = lq_audio / np.abs(lq_audio).max() * 0.85

        lq_path = os.path.join(lq_dir, archivo.name)
        with AudioFile(lq_path, 'w', sr, lq_audio.shape[0]) as f:
            f.write(lq_audio)

        print(f"✅ {archivo.name}")

    return f"✅ {len(archivos)} archivos LQ creados"

import gradio as gr
import torch
import torchaudio
import os

from model import get_model
from dataset import SeminarDataset
from train import train
from inference import enhance_audio
from utils import extraer_zip, verificar_dataset, crear_lq_sintetico

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "/kaggle/working/checkpoints/best_model.pth"
OUTPUT_PATH = "/kaggle/working/audio_mejorado.wav"

print(f"🔥 Device: {DEVICE}")
print(f"🔥 GPUs: {torch.cuda.device_count()}")

# Modelo global
model = get_model(DEVICE)


# ─────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────

def fn_entrenar(zip_file, progress=gr.Progress()):
    global model

    if zip_file is None:
        return "❌ Sube un ZIP primero"

    progress(0, desc="📦 Extrayendo dataset...")
    lq_dir, hq_dir = extraer_zip(zip_file.name)

    errores, n_pares = verificar_dataset(lq_dir, hq_dir)

    if errores:
        return "\n".join(errores)

    progress(0.05, desc=f"✅ {n_pares} pares encontrados")

    try:
        dataset = SeminarDataset(lq_dir, hq_dir)
    except Exception as e:
        return f"❌ Error cargando dataset: {e}"

    model = get_model(DEVICE)

    def callback(pct, msg):
        progress(0.1 + pct * 0.85, desc=f"🔥 {msg}")

    losses, best_loss = train(model, dataset, DEVICE, callback)

    # Recargar mejor checkpoint
    model = get_model(DEVICE, CHECKPOINT)

    progress(1.0, desc="✅ Completado")

    return f"""
✅ ENTRENAMIENTO COMPLETADO

📊 Resultados:
   • Pares usados: {n_pares}
   • Loss inicial: {losses[0]:.6f}
   • Loss final: {losses[-1]:.6f}
   • Mejor loss: {best_loss:.6f}

🎯 Listo para mejorar audios.
"""


def fn_crear_lq(zip_hq, progress=gr.Progress()):
    """Crea LQ sintético a partir de HQ"""

    if zip_hq is None:
        return None, "❌ Sube un ZIP con audios HQ"

    progress(0, desc="📦 Extrayendo...")

    import zipfile, shutil
    destino = "/kaggle/working/solo_hq"
    if os.path.exists(destino):
        shutil.rmtree(destino)
    os.makedirs(destino)

    with zipfile.ZipFile(zip_hq.name) as z:
        z.extractall(destino)

    hq_dir = "/kaggle/working/solo_hq"
    lq_dir = "/kaggle/working/lq_sintetico"

    progress(0.3, desc="🔧 Creando versiones tipo seminario...")
    resultado = crear_lq_sintetico(hq_dir, lq_dir)

    if "❌" in resultado:
        return None, resultado

    # Crear ZIP con los pares
    progress(0.8, desc="📦 Empaquetando dataset...")

    import zipfile as zf
    zip_output = "/kaggle/working/dataset_listo.zip"

    with zf.ZipFile(zip_output, 'w') as z:
        for f in os.listdir(hq_dir):
            if f.endswith('.wav'):
                z.write(os.path.join(hq_dir, f), f"hq/{f}")
        for f in os.listdir(lq_dir):
            if f.endswith('.wav'):
                z.write(os.path.join(lq_dir, f), f"lq/{f}")

    progress(1.0, desc="✅ ZIP listo")

    return zip_output, f"✅ Dataset creado y listo para entrenar"


def fn_mejorar(audio_path, progress=gr.Progress()):
    global model

    if audio_path is None:
        return None, "❌ Sube un audio"

    if not os.path.exists(CHECKPOINT):
        return None, "❌ Primero entrena el modelo"

    model = get_model(DEVICE, CHECKPOINT)

    def callback(pct, msg):
        progress(0.1 + pct * 0.8, desc=f"✨ {msg}")

    progress(0.05, desc="🎵 Procesando...")

    final, sr = enhance_audio(model, audio_path, DEVICE, callback)

    torchaudio.save(OUTPUT_PATH, final.unsqueeze(0), sr)

    progress(1.0, desc="✅ ¡Audio mejorado!")

    duracion = len(final) / sr

    return OUTPUT_PATH, f"""
✅ AUDIO MEJORADO

📊 Info:
   • Duración: {duracion:.1f}s ({duracion/60:.1f} min)
   • Sample rate: {sr}Hz
   • Device usado: {DEVICE}
"""


# ─────────────────────────────────────────
# INTERFAZ
# ─────────────────────────────────────────

with gr.Blocks(
    title="🎙️ Seminario → Estudio",
    theme=gr.themes.Soft(primary_hue="purple")
) as demo:

    gr.Markdown("""
    # 🎙️ Seminario → Estudio
    ### El upscaler que aprende la acústica de TU salón
    ---
    """)

    with gr.Tab("⚗️ 0. Crear Dataset Sintético"):
        gr.Markdown("""
        ### ¿No tienes pares LQ/HQ?
        Sube tus audios limpios y este tab crea las versiones
        tipo seminario automáticamente.
        """)
        zip_hq_input = gr.File(
            label="ZIP con audios limpios (HQ)",
            file_types=[".zip"]
        )
        crear_btn = gr.Button("🔧 Crear Dataset", variant="primary")
        zip_output = gr.File(label="📦 Dataset listo para descargar")
        crear_status = gr.Textbox(label="Estado", interactive=False)

        crear_btn.click(
            fn_crear_lq,
            inputs=[zip_hq_input],
            outputs=[zip_output, crear_status]
        )

    with gr.Tab("🔥 1. Entrenar"):
        gr.Markdown("""
        ### Sube tu dataset de pares LQ/HQ
        ```
        dataset.zip
        ├── lq/  ← audios con eco/ruido
        └── hq/  ← los mismos audios limpios
        ```
        """)
        zip_input = gr.File(
            label="ZIP con dataset (lq/ y hq/)",
            file_types=[".zip"]
        )
        train_btn = gr.Button(
            "🧠 ENTRENAR",
            variant="primary",
            size="lg"
        )
        train_status = gr.Textbox(
            label="Estado del entrenamiento",
            lines=10,
            interactive=False
        )

        train_btn.click(
            fn_entrenar,
            inputs=[zip_input],
            outputs=[train_status]
        )

    with gr.Tab("✨ 2. Mejorar Audio"):
        gr.Markdown("""
        ### Sube tu seminario completo
        Cualquier duración. Se procesa en chunks automáticamente.
        """)
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="🎤 Audio del seminario",
                    type="filepath"
                )
                enhance_btn = gr.Button(
                    "✨ MEJORAR",
                    variant="primary",
                    size="lg"
                )
            with gr.Column():
                audio_output = gr.Audio(
                    label="🔊 Audio mejorado"
                )
                enhance_status = gr.Textbox(
                    label="Estado",
                    lines=8,
                    interactive=False
                )

        enhance_btn.click(
            fn_mejorar,
            inputs=[audio_input],
            outputs=[audio_output, enhance_status]
        )

demo.launch(share=True, debug=True)

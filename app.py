import gradio as gr
import torch
import torchaudio
import os

from model import get_model
from dataset import SeminarDataset
from train import train
from inference import enhance_audio
from utils import (
    extraer_zip,
    verificar_dataset,
    crear_lq_sintetico,
    descargar_desde_hf
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "/kaggle/working/checkpoints/best_model.pth"
OUTPUT_PATH = "/kaggle/working/audio_mejorado.wav"
DATASET_PATH = "/kaggle/working/dataset"

print(f"🔥 Device: {DEVICE}")
print(f"🔥 GPUs: {torch.cuda.device_count()}")

model = get_model(DEVICE)


def fn_descargar_hf(hf_repo, hf_filename, hf_token, progress=gr.Progress()):
    """
    Descarga el ZIP desde Hugging Face y lo extrae.
    hf_repo: "tu_usuario/dataset-seminario"
    hf_filename: "dataset.zip"
    hf_token: tu token de HF (solo si el dataset es privado)
    """

    if not hf_repo or not hf_filename:
        return "❌ Rellena el repo y el nombre del archivo"

    progress(0, desc="⬇️ Conectando con Hugging Face...")

    try:
        token = hf_token if hf_token.strip() != "" else None

        zip_path = descargar_desde_hf(
            repo_id=hf_repo.strip(),
            filename=hf_filename.strip(),
            local_path="/kaggle/working/hf_download",
            token=token
        )

        progress(0.5, desc="📦 Extrayendo ZIP...")

        lq_dir, hq_dir = extraer_zip(
            zip_path,
            destino=DATASET_PATH
        )

        errores, n_pares = verificar_dataset(lq_dir, hq_dir)

        if errores:
            return "❌ Dataset descargado pero con errores:\n" + "\n".join(errores)

        progress(1.0, desc="✅ Dataset listo")

        return f"""
✅ DATASET DESCARGADO Y LISTO

📊 Info:
   • Repo: {hf_repo}
   • Archivo: {hf_filename}
   • Pares encontrados: {n_pares}
   • LQ: {lq_dir}
   • HQ: {hq_dir}

🎯 Ahora ve al tab "Entrenar" y pulsa el botón.
   No necesitas subir ningún ZIP.
"""

    except Exception as e:
        return f"❌ Error descargando: {str(e)}"


def fn_entrenar(zip_file, progress=gr.Progress()):
    global model

    # Primero verificar si ya hay dataset descargado desde HF
    lq_preload = os.path.join(DATASET_PATH, "lq")
    hq_preload = os.path.join(DATASET_PATH, "hq")

    if os.path.exists(lq_preload) and os.path.exists(hq_preload):
        lq_dir = lq_preload
        hq_dir = hq_preload
        progress(0.05, desc="✅ Usando dataset ya descargado desde HuggingFace")

    elif zip_file is not None:
        progress(0, desc="📦 Extrayendo ZIP...")
        lq_dir, hq_dir = extraer_zip(zip_file.name)

    else:
        return (
            "❌ No hay dataset.\n"
            "Opciones:\n"
            "1. Descarga desde HuggingFace en el tab '⬇️ Descargar Dataset'\n"
            "2. Sube un ZIP aquí"
        )

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
    model = get_model(DEVICE, CHECKPOINT)

    progress(1.0, desc="✅ Completado")

    return f"""
✅ ENTRENAMIENTO COMPLETADO

📊 Resultados:
   • Pares usados: {n_pares}
   • Loss inicial: {losses[0]:.6f}
   • Loss final:   {losses[-1]:.6f}
   • Mejor loss:   {best_loss:.6f}

🎯 Listo para mejorar audios.
"""


def fn_crear_lq(zip_hq, progress=gr.Progress()):
    if zip_hq is None:
        return None, "❌ Sube un ZIP con audios HQ"

    import zipfile
    import shutil

    progress(0, desc="📦 Extrayendo...")

    destino = "/kaggle/working/solo_hq"
    if os.path.exists(destino):
        shutil.rmtree(destino)
    os.makedirs(destino)

    with zipfile.ZipFile(zip_hq.name) as z:
        z.extractall(destino)

    hq_dir = destino
    lq_dir = "/kaggle/working/lq_sintetico"

    progress(0.3, desc="🔧 Creando versiones tipo seminario...")
    resultado = crear_lq_sintetico(hq_dir, lq_dir)

    if "❌" in resultado:
        return None, resultado

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
    return zip_output, "✅ Dataset creado y listo para entrenar"


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
   • Device: {DEVICE}
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

    # ── TAB 0: Descargar desde HF ──
    with gr.Tab("⬇️ Descargar Dataset"):
        gr.Markdown("""
        ### Descarga tu dataset desde Hugging Face
        
        Sube tu ZIP a un dataset de HuggingFace y ponlo aquí.
        Así no importa el tamaño del archivo ni tu wifi.
        
        **Cómo subir a HuggingFace:**
        ```
        1. Ve a huggingface.co
        2. Tu perfil → New Dataset
        3. Sube tu ZIP
        4. Copia el nombre del repo (usuario/nombre-dataset)
        ```
        """)

        with gr.Row():
            with gr.Column():
                hf_repo = gr.Textbox(
                    label="Repo de HuggingFace",
                    placeholder="tu_usuario/dataset-seminario",
                    info="El nombre del dataset en HuggingFace"
                )
                hf_filename = gr.Textbox(
                    label="Nombre del archivo ZIP",
                    placeholder="dataset.zip",
                    info="El nombre exacto del archivo que subiste"
                )
                hf_token = gr.Textbox(
                    label="Token de HuggingFace (solo si es privado)",
                    placeholder="hf_xxxxxxxxxxxx",
                    type="password",
                    info="Déjalo vacío si el dataset es público"
                )
                download_btn = gr.Button(
                    "⬇️ DESCARGAR DATASET",
                    variant="primary",
                    size="lg"
                )

            with gr.Column():
                download_status = gr.Textbox(
                    label="Estado de la descarga",
                    lines=12,
                    interactive=False
                )

        download_btn.click(
            fn_descargar_hf,
            inputs=[hf_repo, hf_filename, hf_token],
            outputs=[download_status]
        )

    # ── TAB 1: Crear Dataset Sintético ──
    with gr.Tab("⚗️ Crear Dataset Sintético"):
        gr.Markdown("""
        ### ¿No tienes pares LQ/HQ?
        Sube tus audios limpios y este tab crea
        las versiones tipo seminario automáticamente.
        """)
        zip_hq_input = gr.File(
            label="ZIP con audios limpios (HQ)",
            file_types=[".zip"]
        )
        crear_btn = gr.Button(
            "🔧 Crear Dataset",
            variant="primary"
        )
        zip_output_file = gr.File(
            label="📦 Dataset listo para descargar"
        )
        crear_status = gr.Textbox(
            label="Estado",
            interactive=False
        )

        crear_btn.click(
            fn_crear_lq,
            inputs=[zip_hq_input],
            outputs=[zip_output_file, crear_status]
        )

    # ── TAB 2: Entrenar ──
    with gr.Tab("🔥 Entrenar"):
        gr.Markdown("""
        ### Entrena el modelo con tu dataset
        
        Si ya descargaste desde HuggingFace, 
        pulsa directamente el botón sin subir ZIP.
        
        Si no, sube tu ZIP aquí.
        
        ```
        dataset.zip
        ├── lq/  ← audios con eco/ruido
        └── hq/  ← los mismos audios limpios
        ```
        """)
        zip_input = gr.File(
            label="ZIP con dataset (opcional si ya descargaste de HF)",
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

    # ── TAB 3: Mejorar Audio ──
    with gr.Tab("✨ Mejorar Audio"):
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

demo.launch(
    share=True,
    debug=True,
    max_file_size="500mb"
)

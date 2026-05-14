import gradio as gr
import torch
import os

from model import get_model
from dataset import MusicDataset
from train import train
from inference import enhance_audio
from utils import extraer_zip, verificar_dataset, descargar_desde_hf

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT   = "/kaggle/working/checkpoints/best_model.pth"
DATASET_PATH = "/kaggle/working/dataset"

print(f"🔥 Device: {DEVICE}")
print(f"🔥 GPUs:   {torch.cuda.device_count()}")

model = get_model(DEVICE)


def fn_descargar_hf(hf_url, hf_token, progress=gr.Progress()):
    if not hf_url or hf_url.strip() == "":
        return "❌ Pon el link completo"

    progress(0, desc="⬇️ Conectando...")

    try:
        token    = hf_token.strip() if hf_token.strip() != "" else None
        progress(0.1, desc="⬇️ Descargando...")

        zip_path = descargar_desde_hf(
            url        = hf_url.strip(),
            local_path = "/kaggle/working/hf_download",
            token      = token
        )

        progress(0.7, desc="📦 Extrayendo...")
        lq_dir, hq_dir = extraer_zip(zip_path, destino=DATASET_PATH)
        errores, n     = verificar_dataset(lq_dir, hq_dir)

        if errores:
            return "❌ Errores:\n" + "\n".join(errores)

        progress(1.0, desc="✅ Listo")
        return f"✅ Dataset listo: {n} pares\n🎯 Ve al tab Entrenar."

    except Exception as e:
        return f"❌ Error: {str(e)}"


def fn_entrenar(zip_file, progress=gr.Progress()):
    global model

    lq_pre = os.path.join(DATASET_PATH, "lq")
    hq_pre = os.path.join(DATASET_PATH, "hq")

    if os.path.exists(lq_pre) and os.path.exists(hq_pre):
        lq_dir = lq_pre
        hq_dir = hq_pre
        progress(0.02, desc="✅ Dataset encontrado")

    elif zip_file is not None:
        progress(0, desc="📦 Extrayendo...")
        lq_dir, hq_dir = extraer_zip(zip_file.name)

    else:
        return "❌ No hay dataset."

    errores, n_pares = verificar_dataset(lq_dir, hq_dir)
    if errores:
        return "\n".join(errores)

    progress(0.05, desc=f"✅ {n_pares} pares listos")

    try:
        dataset = MusicDataset(lq_dir, hq_dir)
    except Exception as e:
        return f"❌ Error cargando dataset: {e}"

    model = get_model(DEVICE)

    def callback(pct, msg):
        progress(0.05 + pct * 0.9, desc=f"🔥 {msg}")

    losses, best_loss = train(model, dataset, DEVICE, callback)

    model = get_model(DEVICE, checkpoint_path=CHECKPOINT)
    progress(1.0, desc="✅ Completado")

    return f"""
✅ ENTRENAMIENTO COMPLETADO

📊 Resultados:
   • Pares:        {n_pares}
   • Loss inicial: {losses[0]:.6f}
   • Loss final:   {losses[-1]:.6f}
   • Mejor loss:   {best_loss:.6f}

🎯 Modelo listo para restaurar.
"""


def fn_mejorar(audio_path, progress=gr.Progress()):
    global model

    if audio_path is None:
        return None, "❌ Sube un audio"

    if not os.path.exists(CHECKPOINT):
        return None, "❌ Primero entrena el modelo"

    model = get_model(DEVICE, checkpoint_path=CHECKPOINT)

    def callback(pct, msg):
        progress(pct, desc=f"✨ {msg}")

    try:
        out, sr = enhance_audio(model, audio_path, DEVICE, callback)
        progress(1.0, desc="✅ Listo")
        return out, "✅ Canción restaurada en PCM 24-bit"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


with gr.Blocks(
    title="🎵 Music Restorer",
    theme=gr.themes.Soft(primary_hue="purple")
) as demo:

    gr.Markdown("""
    # 🎵 Music Restorer
    ### Restaura canciones completas grabadas en calidad seminario
    ---
    """)

    with gr.Tab("⬇️ Descargar Dataset"):
        gr.Markdown("### Descarga tu dataset desde HuggingFace")
        with gr.Row():
            with gr.Column():
                hf_url = gr.Textbox(
                    label       = "Link directo del ZIP",
                    placeholder = "https://huggingface.co/datasets/usuario/repo/resolve/main/dataset.zip"
                )
                hf_token = gr.Textbox(
                    label = "Token HF (solo si es privado)",
                    type  = "password"
                )
                download_btn = gr.Button(
                    "⬇️ DESCARGAR",
                    variant="primary"
                )
            with gr.Column():
                download_status = gr.Textbox(
                    label       = "Estado",
                    lines       = 8,
                    interactive = False
                )
        download_btn.click(
            fn_descargar_hf,
            inputs  = [hf_url, hf_token],
            outputs = [download_status]
        )

    with gr.Tab("🔥 Entrenar"):
        gr.Markdown("""
        ### Entrena con tus pares LQ/HQ
        ```
        dataset.zip
        ├── lq/  ← canciones calidad seminario
        └── hq/  ← las mismas canciones en limpio
        ```
        """)
        zip_input = gr.File(
            label      = "ZIP (opcional si ya descargaste de HF)",
            file_types = [".zip"]
        )
        train_btn    = gr.Button("🧠 ENTRENAR", variant="primary", size="lg")
        train_status = gr.Textbox(
            label       = "Estado",
            lines       = 12,
            interactive = False
        )
        train_btn.click(
            fn_entrenar,
            inputs  = [zip_input],
            outputs = [train_status]
        )

    with gr.Tab("✨ Restaurar Canción"):
        gr.Markdown("### Sube la canción en calidad seminario")
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label = "🎵 Canción LQ",
                    type  = "filepath"
                )
                enhance_btn = gr.Button(
                    "✨ RESTAURAR",
                    variant = "primary",
                    size    = "lg"
                )
            with gr.Column():
                audio_output   = gr.Audio(label="🔊 Canción restaurada")
                enhance_status = gr.Textbox(
                    label       = "Estado",
                    lines       = 6,
                    interactive = False
                )
        enhance_btn.click(
            fn_mejorar,
            inputs  = [audio_input],
            outputs = [audio_output, enhance_status]
        )

demo.launch(
    share         = True,
    debug         = True,
    max_file_size = "500mb"
)

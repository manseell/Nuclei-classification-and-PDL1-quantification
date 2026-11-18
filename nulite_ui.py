"""
NuLite Nuclei Detection — Interfaccia Gradio (compatibile Gradio 6+)
Requisiti: pip install gradio Pillow

Avvio: python3 nulite_ui.py
Poi apri nel browser: http://localhost:7860
"""

import gradio as gr
import subprocess
import os
import glob
import json
import sys
import shutil
from pathlib import Path


# ── Configurazione percorsi (modifica qui) ──────────────────────────────────
IMAGES_DIR         = "/home/jovyan/path-vol/Flusso1/NuLite/Training_1_5_26/Prova_png_HEM_only/patches/"
RGB_DIR            = "/home/jovyan/path-vol/Flusso1/NuLite/Training_1_5_26/Prova_png_norm_blues/patches/"
MODEL_PATH         = "./logging/checkpoints/modeldef_21_5.pth"
SCRIPTS_DIR        = "."
UPLOAD_DIR         = "./Upload/"
RADIAL_SCRIPT_PATH = "./radial_suppression.py"
QCS_SCRIPT_PATH    = "./QCS_radial_suppression.py"

# GeoJSON predefinito per Step 3 (Render)
CELLS_GEOJSON          = "/home/jovyan/path-vol/Flusso1/NuLite/Training_1_5_26/Prova_png_HEM_only/patches/cell_detection/cells.geojson"
# GeoJSON predefinito per Step 4 (Radial Suppression)
CELL_DETECTION_GEOJSON = "/home/jovyan/path-vol/Flusso1/NuLite/Training_1_5_26/Prova_png_HEM_only/patches/cell_detection/cell_detection.geojson"

# stato condiviso tra le funzioni
_state = {}
# ────────────────────────────────────────────────────────────────────────────


def get_image_list():
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.svs")
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(IMAGES_DIR, "**", ext), recursive=True)
        files += glob.glob(os.path.join(IMAGES_DIR, ext))
    return sorted(set(files))


def run_cmd(cmd, cwd=None):
    log = f"$ {' '.join(cmd)}\n\n"
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd or SCRIPTS_DIR,
            timeout=600,
        )
        log += result.stdout or ""
        ok = result.returncode == 0
        log += f"\n{'✅ Completato con successo.' if ok else f'❌ Exit code {result.returncode}'}"
        return log, ok
    except FileNotFoundError as e:
        return log + f"\n❌ File non trovato: {e}", False
    except subprocess.TimeoutExpired:
        return log + "\n❌ Timeout superato (600s).", False
    except Exception as e:
        return log + f"\n❌ Errore: {e}", False


# ── Fix ICC profile ──────────────────────────────────────────────────────────
def fix_icc_profile(image_path):
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        Image.open(image_path).convert("RGB")
        return image_path, ""
    except (ValueError, Exception):
        pass
    try:
        import io
        with open(image_path, "rb") as f:
            raw = f.read()
        img = Image.open(io.BytesIO(raw))
        img.info.pop("icc_profile", None)
        img = img.convert("RGB")
        tmp_path = f"/tmp/nulite_fixed_{Path(image_path).name}"
        img.save(tmp_path, format="PNG")
        return tmp_path, f"⚠️ Profilo ICC rimosso automaticamente.\n   Originale: {image_path}\n   Usato:     {tmp_path}\n\n"
    except Exception as e:
        return image_path, f"⚠️ Fix ICC fallito ({e}), uso originale.\n\n"


# ── Gestione upload immagine ─────────────────────────────────────────────────
def handle_upload(uploaded_file, current_choices):
    """
    Copia il file caricato in IMAGES_DIR e lo aggiunge al dropdown.
    Restituisce (nuovo valore dropdown, nuova lista choices, messaggio).
    """
    if uploaded_file is None:
        return gr.update(), gr.update(), "⚠️ Nessun file caricato."

    dest_path = os.path.join(UPLOAD_DIR, Path(uploaded_file).name)
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        shutil.copy2(uploaded_file, dest_path)
        new_list = get_image_list()
        return (
            gr.update(choices=new_list, value=dest_path),
            f"✅ Immagine copiata in:\n{dest_path}",
        )
    except Exception as e:
        return gr.update(), f"❌ Errore durante il caricamento: {e}"


# ── Step 2: Inference ────────────────────────────────────────────────────────
def run_inference(image_path, model_path):
    if not image_path:
        return "⚠️ Nessuna immagine selezionata.", ""
    image_path = image_path.strip()
    model_path = model_path.strip()
    if not os.path.isfile(image_path):
        return f"❌ Immagine non trovata:\n{image_path}", ""
    if not os.path.isfile(model_path):
        return f"❌ Modello non trovato:\n{model_path}", ""

    _state["image_path"] = image_path
    image_path, icc_note = fix_icc_profile(image_path)

    cmd = [
        sys.executable, "inference_patch.py",
        "--model", model_path,
        "--image", image_path,
        "--gpu", "0",
        "--geojson",
        "--magnification", "20",
    ]
    log, ok = run_cmd(cmd)
    log = icc_note + log

    geojson_found = ""
    if ok:
        stem = Path(image_path).stem
        found = []
        for p in [
            os.path.join(SCRIPTS_DIR, f"{stem}*.geojson"),
            os.path.join(SCRIPTS_DIR, "output", f"{stem}*.geojson"),
            os.path.join(SCRIPTS_DIR, "**", f"{stem}*.geojson"),
        ]:
            found += glob.glob(p, recursive=True)
        if found:
            geojson_found = max(found, key=os.path.getmtime)
            log += f"\n\n🗺️ GeoJSON rilevato:\n{geojson_found}"

    return log, CELLS_GEOJSON


# ── Step 3: Render ───────────────────────────────────────────────────────────
def run_render(image_path, geojson_path):
    if not image_path:
        return "⚠️ Nessuna immagine selezionata.", None, ""
    geojson_path = (geojson_path or "").strip()
    if not geojson_path or not os.path.isfile(geojson_path):
        return f"❌ File GeoJSON non trovato:\n{geojson_path}", None, ""

    _state["image_path"]   = image_path.strip()
    _state["geojson_path"] = geojson_path

    cmd = [
        sys.executable, "render_geojson.py",
        "--image",   image_path.strip(),
        "--geojson", geojson_path,
    ]
    log, ok = run_cmd(cmd)

    output_img = None
    stats_text = ""

    if ok:
        img_dir      = str(Path(image_path.strip()).parent)
        fixed_output = os.path.join(img_dir, "patch_inference_result_boundaries.png")
        stem         = Path(image_path.strip()).stem
        candidates   = [fixed_output] + [
            p for pat in [
                os.path.join(img_dir, f"{stem}*render*.png"),
                os.path.join(SCRIPTS_DIR, "patch_inference_result_boundaries.png"),
                os.path.join(SCRIPTS_DIR, f"{stem}*.png"),
            ]
            for p in glob.glob(pat, recursive=True)
        ]
        for c in candidates:
            if os.path.isfile(c):
                output_img = c
                log += f"\n\n🖼️ Immagine caricata:\n{c}"
                break

        stats_lines = [
            line for line in log.splitlines()
            if any(k in line for k in ("Totale nuclei", "nuclei", "Output salvato"))
        ]
        stats_text = "\n".join(stats_lines) if stats_lines else ""

        if not stats_text:
            json_path = geojson_path.replace(".geojson", ".json")
            if os.path.isfile(json_path):
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                    stats_text = "\n".join(
                        f"• {k}: {v}" for k, v in data.items()
                    ) if isinstance(data, dict) else json.dumps(data, indent=2)
                except Exception:
                    pass

    return log, output_img, stats_text


# ── Step 4: Radial Suppression ───────────────────────────────────────────────
def run_notebook(script_path, radial_geojson_path, soglia_dab, soglia_tps, n_raggi):
    script_path         = script_path.strip()
    radial_geojson_path = radial_geojson_path.strip()

    if not os.path.isfile(script_path):
        return "❌ Script non trovato.", None, ""

    img_blue = _state.get("image_path", "")
    if not img_blue or not os.path.isfile(img_blue):
        return "❌ Seleziona un'immagine valida prima di eseguire la radial suppression.", None, ""

    if not radial_geojson_path or not os.path.isfile(radial_geojson_path):
        return f"❌ File GeoJSON per radial suppression non trovato:\n{radial_geojson_path}", None, ""

    # Switch automatico da immagine blu a immagine RGB
    rel_path = os.path.relpath(img_blue, IMAGES_DIR)
    img_rgb  = os.path.join(RGB_DIR, rel_path)

    if not os.path.isfile(img_rgb):
        return f"❌ Immagine RGB corrispondente non trovata in RGB_DIR:\n{img_rgb}", None, ""

    out_dir = os.path.join(SCRIPTS_DIR, "radial_output")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, script_path,
        "--image",      img_rgb,
        "--geojson",    radial_geojson_path,
        "--output_dir", out_dir,
        "--soglia_dab", str(round(soglia_dab, 4)),
        "--soglia_tps", str(round(soglia_tps, 4)),
        "--n_raggi", str(round(n_raggi, 4)),
    ]

    log_init = (
        f"ℹ️ Switch immagine effettuato.\n"
        f"   Input Blu : {img_blue}\n"
        f"   Input RGB : {img_rgb}\n"
        f"   Soglia DAB: {soglia_dab}\n"
        f"   Soglia TPS: {soglia_tps}\n"
        f"   Numero raggi: {n_raggi}\n\n"
    )
    log, ok = run_cmd(cmd, cwd=str(Path(script_path).parent))
    log = log_init + log

    output_img = None
    stats_text = ""

    if ok:
        main_out = os.path.join(out_dir, "radial_suppression_result.png")
        if os.path.isfile(main_out):
            output_img = main_out
            log += f"\n\n🖼️ Immagine caricata:\n{main_out}"

        stats_lines = [
            line for line in log.splitlines()
            if any(k in line for k in ("Cellule", "TPS", "═", "positive", "negative"))
        ]
        stats_text = "\n".join(stats_lines)

    return log, output_img, stats_text


# ── Step 5: QCS Radial Suppression ───────────────────────────────────────────────
def run_qcs(script_path, radial_geojson_path, soglia_dab, soglia_bassa, soglia_media, soglia_alta, n_raggi):
    qcs_script_path         = script_path.strip()
    radial_geojson_path = radial_geojson_path.strip()

    if not os.path.isfile(qcs_script_path):
        return "❌ Script non trovato.", None, ""

    img_blue = _state.get("image_path", "")
    if not img_blue or not os.path.isfile(img_blue):
        return "❌ Seleziona un'immagine valida prima di eseguire la radial suppression.", None, ""

    if not radial_geojson_path or not os.path.isfile(radial_geojson_path):
        return f"❌ File GeoJSON per radial suppression non trovato:\n{radial_geojson_path}", None, ""

    # Switch automatico da immagine blu a immagine RGB
    rel_path = os.path.relpath(img_blue, IMAGES_DIR)
    img_rgb  = os.path.join(RGB_DIR, rel_path)

    if not os.path.isfile(img_rgb):
        return f"❌ Immagine RGB corrispondente non trovata in RGB_DIR:\n{img_rgb}", None, ""

    out_dir = os.path.join(SCRIPTS_DIR, "qcs_output")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, qcs_script_path,
        "--image",      img_rgb,
        "--geojson",    radial_geojson_path,
        "--output_dir", out_dir,
        "--soglia_dab", str(round(soglia_dab, 4)),
        "--soglia_bassa", str(round(soglia_bassa, 4)),
        "--soglia_media", str(round(soglia_media, 4)),
        "--soglia_alta", str(round(soglia_alta, 4)),
        "--n_raggi", str(round(n_raggi, 4)),
    ]

    log_init = (
        f"ℹ️ Switch immagine effettuato.\n"
        f"   Input Blu : {img_blue}\n"
        f"   Input RGB : {img_rgb}\n"
        f"   Soglia DAB: {soglia_dab}\n"
        f"   Soglia bassa: {soglia_bassa}\n"
        f"   Soglia media: {soglia_media}\n"
        f"   Soglia alta: {soglia_alta}\n"
        f"   Numero raggi: {n_raggi}\n\n"
    )
    log, ok = run_cmd(cmd, cwd=str(Path(script_path).parent))
    log = log_init + log

    output_img = None
    stats_text = ""

    if ok:
        main_out = os.path.join(out_dir, "radial_suppression_result_qcs.png")
        if os.path.isfile(main_out):
            output_img = main_out
            log += f"\n\n🖼️ Immagine caricata:\n{main_out}"

        stats_lines = [
            line for line in log.splitlines()
            if any(k in line for k in ("Cellule", "H-score", "PMSTC", "="))
        ]
        stats_text = "\n".join(stats_lines)

    return log, output_img, stats_text


# ── UI ────────────────────────────────────────────────────────────────────────
CSS = """
#title { font-family: monospace; font-size: 2rem; font-weight: bold; color: #00C6A2; margin-bottom: 0; }
#subtitle { font-family: monospace; color: #7C82A0; margin-top: 2px; margin-bottom: 1rem; }
footer { display: none !important; }
"""

def build_ui():
    image_list = get_image_list()

    with gr.Blocks(title="NuLite — Nuclei Detection") as demo:

        gr.HTML('<p id="title">QUANTIFICAZIONE PD-L1</p><p id="subtitle">Pipeline per la quantificazione del PD-L1 interno a cellule tumorali</p>')

        gr.Markdown("""
        **Funzionamento:**
        
        1. Seleziona l'immagine in 2 modi:
            - Scegli l'immagine dalla lista presente
            - Carica l'immagine da computer, scegliendo l'immagine e poi cliccando su *"Carica e aggiungi a lista"*
        
        2. Esegui l'inferenza cliccando *"Esegui Inference"*
            - Puoi anche selezionare altri checkpoints scrivendo il percorso
            - Alla fine comparirà il percorso del file `.geojson` per il rendering
            - **NB:** Se hai caricato un'immagine, modifica il percorso con:  
              `/home/jovyan/path-vol/Flusso1/NuLite/Upload/cell_detection/cells.geojson`
        
        3. Esegui il render cliccando *"Esegui Render"*
        
        4. Esegui il calcolo del TPS cliccando *"Calcolo TPS"*
            - È possibile selezionare la soglia DAB
            - È possibile selezionare la soglia TPS
            - È possibile modificare il percorso del file `.geojson`
            - **NB:** Se hai caricato un'immagine, modifica il percorso con:  
              `/home/jovyan/path-vol/Flusso1/NuLite/Upload/cell_detection/cell_detection.geojson`

        5. Esegui il calcolo del QCS cliccando *"Calcolo QCS"*
            - È possibile selezionare la soglia DAB
            - È possibile selezionare la soglia bassa, la soglia media e la soglia alta
            - È possibile modificare il percorso del file `.geojson`
            - **NB:** Se hai caricato un'immagine, modifica il percorso con:  
              `/home/jovyan/path-vol/Flusso1/NuLite/Upload/cell_detection/cell_detection.geojson`
        """)

        # ── Step 1 ────────────────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### ① Selezione immagine")

            with gr.Row():
                image_dd = gr.Dropdown(
                    choices=image_list,
                    label="Scegli dalla lista",
                    value=image_list[0] if image_list else None,
                    allow_custom_value=True,
                    scale=4,
                )
                refresh_btn = gr.Button("↺  Aggiorna lista", scale=1)

            with gr.Row():
                upload_file = gr.File(
                    label="Oppure carica un'immagine dal tuo PC",
                    file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                    scale=4,
                )
                upload_btn  = gr.Button("⬆  Carica e aggiungi alla lista", scale=1)

            upload_msg = gr.Textbox(label="Stato upload", lines=2, interactive=False)

        refresh_btn.click(
            fn=lambda: gr.update(choices=get_image_list()),
            outputs=image_dd,
        )
        upload_btn.click(
            fn=handle_upload,
            inputs=[upload_file, image_dd],
            outputs=[image_dd, upload_msg],
        )

        # ── Step 2 ────────────────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### ② Inference  —  `inference_patch.py`")
            with gr.Row():
                model_tb = gr.Textbox(
                    value=MODEL_PATH,
                    label="Percorso modello .pth",
                    scale=4,
                )
                inf_btn = gr.Button("▶  Esegui Inference", variant="primary", scale=1)

            inf_log = gr.Textbox(label="Log inference", lines=6, interactive=False)
            geojson_tb = gr.Textbox(
                label="GeoJSON per Render (cells.geojson)",
                value=CELLS_GEOJSON,
            )

        inf_btn.click(
            fn=run_inference,
            inputs=[image_dd, model_tb],
            outputs=[inf_log, geojson_tb],
        )

        # ── Step 3 ────────────────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### ③ Render GeoJSON  —  `render_geojson.py`")
            render_btn = gr.Button("▶  Esegui Render", variant="primary")
            with gr.Row():
                render_log = gr.Textbox(
                    label="Log render", lines=8, interactive=False, scale=1
                )
                with gr.Column(scale=1):
                    render_img   = gr.Image(label="Immagine output", type="filepath")
                    render_stats = gr.Textbox(label="Statistiche", lines=4, interactive=False)

        render_btn.click(
            fn=run_render,
            inputs=[image_dd, geojson_tb],
            outputs=[render_log, render_img, render_stats],
        )

        # ── Step 4 ────────────────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### ④ Radial Suppression  —  `radial_suppression.py`")

            with gr.Row():
                nb_tb = gr.Textbox(
                    value=RADIAL_SCRIPT_PATH,
                    label="Percorso script",
                    scale=2,
                )
                radial_geojson_tb = gr.Textbox(
                    value=CELL_DETECTION_GEOJSON,
                    label="GeoJSON (cell_detection.geojson)",
                    scale=3,
                )

            with gr.Row():
                soglia_dab_sl = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01,
                    value=0.10,
                    label="Soglia DAB  (intensità minima picco)",
                    scale=1,
                )
                soglia_tps_sl = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01,
                    value=0.50,
                    label="Soglia TPS  (frazione raggi positivi)",
                    scale=1,
                )
                raggi_sl = gr.Slider(
                    minimum=0, maximum=100, step=1,
                    value=36,
                    label="Numero raggi",
                    scale=1,
                )
                nb_btn = gr.Button("▶  Calcolo TPS", variant="primary", scale=1)

            with gr.Row():
                nb_log = gr.Textbox(label="Log", lines=8, interactive=False, scale=1)
                with gr.Column(scale=1):
                    nb_img   = gr.Image(label="Immagine output", type="filepath")
                    nb_stats = gr.Textbox(label="Output testuale", lines=4, interactive=False)

        nb_btn.click(
            fn=run_notebook,
            inputs=[nb_tb, radial_geojson_tb, soglia_dab_sl, soglia_tps_sl, raggi_sl],
            outputs=[nb_log, nb_img, nb_stats],
        )

        # ── Step 5 ────────────────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### ⑤ QCS Radial Suppression  —  `QCS_radial_suppression.py`")

            with gr.Row():
                qcs_nb_tb = gr.Textbox(
                    value=QCS_SCRIPT_PATH,
                    label="Percorso script",
                    scale=2,
                )
                qcs_geojson_tb = gr.Textbox(
                    value=CELL_DETECTION_GEOJSON,
                    label="GeoJSON (cell_detection.geojson)",
                    scale=3,
                )

            with gr.Row():
                qcs_dab_sl = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01,
                    value=0.10,
                    label="Soglia DAB  (intensità minima picco)",
                    scale=1,
                )
                soglia_bassa_sl = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01,
                    value=0.05,
                    label="Soglia bassa",
                    scale=1,
                )
                soglia_media_sl = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01,
                    value=0.10,
                    label="Soglia media",
                    scale=1,
                )
                soglia_alta_sl = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01,
                    value=0.20,
                    label="Soglia alta",
                    scale=1,
                )
                raggi_sl = gr.Slider(
                    minimum=0, maximum=100, step=1,
                    value=36,
                    label="Numero raggi",
                    scale=1,
                )
                
                qcs_btn = gr.Button("▶  Calcolo QCS", variant="primary", scale=1)

            with gr.Row():
                qcs_log = gr.Textbox(label="Log", lines=8, interactive=False, scale=1)
                with gr.Column(scale=1):
                    qcs_img   = gr.Image(label="Immagine output", type="filepath")
                    qcs_stats = gr.Textbox(label="Output testuale", lines=4, interactive=False)

        qcs_btn.click(
            fn=run_qcs,
            inputs=[qcs_nb_tb, qcs_geojson_tb, qcs_dab_sl, soglia_bassa_sl, soglia_media_sl, soglia_alta_sl, raggi_sl],
            outputs=[qcs_log, qcs_img, qcs_stats],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=False,
        css=CSS,
    )
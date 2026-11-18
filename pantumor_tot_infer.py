"""
Pipeline per nuclei segmentation con NuLite su immagini più grandi del patch size supportato (1024x1024).

Step:
1. Padding reflect dell'immagine fino a una dimensione "coprita esattamente" da patching con stride fisso.
2. Patching 1024x1024 con overlap (default 64px -> stride 960).
3. Inference per-patch chiamando lo script CLI esistente (inference_patch.py) via subprocess.
4. Merge dei GeoJSON: traslazione coordinate locali -> globali, rimozione duplicati in overlap, rimozione nuclei nel padding.
5. (Opzionale) Se viene passato un CSV di ground truth, calcola precision/recall/F1 di
   detection e un classification report (stesso criterio di matching bbox-centroide
   usato in pantumor_inference.py) confrontando il GeoJSON finale con il GT.

Uso:
    python3 patch_infer_merge.py \
        --image /path/to/slide.tif \
        --output_dir /path/to/output \
        --model ./logging/checkpoints/model_ready.pth \
        --inference_script ./inference_patch.py \
        --gpu 0 \
        --magnification 20 \
        --patch_size 1024 \
        --overlap 64 \
        --gt /path/to/GC_01.csv \
        --dist_thresh 25


    python3 patch_infer_merge.py \
        --image_dir immagini_pantumor/ \
        --output_dir output/pantumor_full \
        --model ./logging/checkpoints/model_ready.pth \
        --inference_script ./inference_patch.py \
        --gpu 0 --magnification 20 \
        --gt_dir csv_gt_pantumor/ \
        --dist_thresh 25
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from shapely.geometry import shape, mapping, Point
    from shapely.affinity import translate
except ImportError:
    print("ERRORE: serve 'shapely'. Installa con: pip install shapely --break-system-packages", file=sys.stderr)
    raise

try:
    from scipy.spatial import cKDTree
except ImportError:
    print("ERRORE: serve 'scipy'. Installa con: pip install scipy --break-system-packages", file=sys.stderr)
    raise

try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# --------------------------------------------------------------------------------------
# 0. Costanti per il calcolo delle metriche (stesse di pantumor_inference.py)
# --------------------------------------------------------------------------------------

GT_CLASS_NAMES = {1: "Neoplastic", 2: "Inflammatory", 3: "Other"}

GEOJSON_NAME_TO_CLASS = {
    "neoplastic": 1,
    "inflammatory": 2,
    "other": 3,
    "cancerous epithelial": 1,
    "normal epithelial": 3,
    "dysplastic epithelial": 1,
    "lymphocyte (small)": 2,
    "plasma cell": 2,
    "neutrophil/eosinophil": 2,
    "macrophage": 2,
    "fibroblast": 3,
    "endothelial": 3,
    "miscellaneous": 3,
    "connective": 3,
    "dead": 3,
    "epithelial": 3
}

CSV_NAME_TO_CLASS = {
    "tumor cell": 1,          # Neoplastic
    "cd3+ immune cell": 2,    # Inflammatory
    "other cell": 3,          # Other
}


# --------------------------------------------------------------------------------------
# 1. Padding reflect
# --------------------------------------------------------------------------------------

def compute_padded_size(original_size: int, patch_size: int, stride: int) -> int:
    """
    Calcola la dimensione minima >= original_size tale che:
        padded_size = patch_size + k * stride   per qualche intero k >= 0
    cioè il patching con questo stride/patch_size copre esattamente padded_size senza
    lasciare un ultimo patch che esce parzialmente fuori (lo gestiamo includendolo per intero).
    """
    if original_size <= patch_size:
        return patch_size
    # quanti stride servono dopo il primo patch per coprire il resto
    remaining = original_size - patch_size
    k = int(np.ceil(remaining / stride))
    return patch_size + k * stride


def pad_image_reflect(img: np.ndarray, patch_size: int, stride: int):
    """
    Applica reflect padding (solo bottom/right, l'origine resta a (0,0)) cosi' il
    patching parte esattamente dall'angolo originale e copre tutta l'immagine.

    Ritorna: (img_padded, pad_bottom, pad_right)
    """
    h, w = img.shape[0], img.shape[1]
    padded_h = compute_padded_size(h, patch_size, stride)
    padded_w = compute_padded_size(w, patch_size, stride)

    pad_bottom = padded_h - h
    pad_right = padded_w - w

    if img.ndim == 3:
        pad_width = ((0, pad_bottom), (0, pad_right), (0, 0))
    else:
        pad_width = ((0, pad_bottom), (0, pad_right))

    img_padded = np.pad(img, pad_width, mode="reflect")
    return img_padded, pad_bottom, pad_right


# --------------------------------------------------------------------------------------
# 2. Patching
# --------------------------------------------------------------------------------------

def generate_patch_coords(padded_h: int, padded_w: int, patch_size: int, stride: int):
    """
    Genera la lista di (y0, x0) angoli top-left dei patch, in coordinate dell'immagine
    PADDATA. Patching regolare: dato che padded_size = patch_size + k*stride, l'ultimo
    angolo valido e' esattamente padded_size - patch_size.
    """
    ys = list(range(0, padded_h - patch_size + 1, stride))
    xs = list(range(0, padded_w - patch_size + 1, stride))
    if ys[-1] != padded_h - patch_size:
        ys.append(padded_h - patch_size)
    if xs[-1] != padded_w - patch_size:
        xs.append(padded_w - patch_size)
    coords = [(y, x) for y in ys for x in xs]
    return coords


def save_patches(img_padded: np.ndarray, coords, patch_size: int, patches_dir: Path):
    """
    Ogni patch viene salvato in una propria sottocartella dedicata
    (patches/patch_y{y0}_x{x0}/patch_y{y0}_x{x0}.png) invece che tutti
    nella stessa cartella. Questo e' necessario perche' inference_patch.py
    salva l'output con un nome FISSO (cell_detection/cells.geojson) relativo
    alla cartella dell'immagine in input: se tutti i patch fossero nella
    stessa cartella, ogni run sovrascriverebbe il risultato del run precedente.
    """
    patches_dir.mkdir(parents=True, exist_ok=True)
    patch_paths = []
    for (y0, x0) in coords:
        tile = img_padded[y0:y0 + patch_size, x0:x0 + patch_size]
        name = f"patch_y{y0}_x{x0}.png"
        patch_subdir = patches_dir / f"patch_y{y0}_x{x0}"
        patch_subdir.mkdir(parents=True, exist_ok=True)
        out_path = patch_subdir / name
        Image.fromarray(tile).save(out_path)
        patch_paths.append((out_path, y0, x0))
    return patch_paths


# --------------------------------------------------------------------------------------
# 3. Inference via CLI esistente
# --------------------------------------------------------------------------------------

def run_inference_on_patch(inference_script: Path, model_path: Path, image_path: Path,
                            gpu: int, magnification: int, extra_args=None):
    """
    Lancia inference_patch.py su un singolo tile. Lo script salva l'output con un
    nome FISSO in una sottocartella accanto all'immagine:
        <cartella_immagine>/cell_detection/cells.geojson
    Per questo motivo ogni patch deve vivere nella sua sottocartella dedicata
    (vedi save_patches), altrimenti run successivi sovrascrivono i precedenti.
    """
    cmd = [
        sys.executable, str(inference_script),
        "--model", str(model_path),
        "--image", str(image_path),
        "--gpu", str(gpu),
        "--geojson",
        "--magnification", str(magnification),
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] Inference fallita su {image_path.name}:")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        return None

    expected_geojson_path = image_path.parent / "cell_detection" / "cells.geojson"
    if not expected_geojson_path.exists():
        # fallback: cerca un qualsiasi .geojson dentro cell_detection/ o accanto all'immagine
        candidates = list((image_path.parent / "cell_detection").glob("*.geojson")) \
            if (image_path.parent / "cell_detection").exists() else []
        if not candidates:
            candidates = list(image_path.parent.glob("*.geojson"))
        if candidates:
            expected_geojson_path = candidates[0]
        else:
            print(f"[WARN] Nessun GeoJSON trovato per {image_path.name} dopo l'inference.")
            return None

    return expected_geojson_path


# --------------------------------------------------------------------------------------
# 4. Merge: traslazione coordinate + rimozione duplicati + rimozione nuclei nel padding
# --------------------------------------------------------------------------------------

def load_geojson_features(geojson_path: Path):
    with open(geojson_path, "r") as f:
        data = json.load(f)

    # inference_patch.py salva una lista di feature direttamente, non una
    # FeatureCollection standard {"type": ..., "features": [...]}.
    # Gestiamo entrambi i casi per robustezza.
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get("features", [])
    else:
        print(f"[WARN] Formato GeoJSON inatteso in {geojson_path}: {type(data)}")
        return []


def explode_features_per_nucleus(features):
    """
    inference_patch.py salva UNA Feature per classe cellulare (es. 'Neoplastic',
    'Inflammatory', 'Other'), e dentro ogni Feature il campo geometry.coordinates
    di tipo MultiPolygon contiene TUTTI i nuclei di quella classe come poligoni
    separati. Quindi 3 Feature totali possono contenere centinaia di nuclei.

    Questa funzione spacchetta ogni Feature-per-classe in N pseudo-feature,
    una per ciascun poligono/nucleo, ereditando le 'properties' (classe) della
    Feature originale. Da qui in avanti nel codice, "feature" = un singolo nucleo.
    """
    exploded = []
    for feat in features:
        geom_type = feat.get("geometry", {}).get("type")
        coords = feat.get("geometry", {}).get("coordinates", [])
        props = feat.get("properties", {})
        base_id = feat.get("id", "")

        if geom_type == "MultiPolygon":
            # coords e' una lista di poligoni; ciascuno e' un singolo nucleo
            for i, poly_coords in enumerate(coords):
                exploded.append({
                    "type": "Feature",
                    "id": f"{base_id}_{i}",
                    "geometry": {"type": "Polygon", "coordinates": poly_coords},
                    "properties": props,
                })
        elif geom_type == "Polygon":
            # già un singolo nucleo
            exploded.append(feat)
        else:
            print(f"[WARN] Tipo di geometria inatteso durante l'esplosione: {geom_type}")

    return exploded


def translate_feature_geometry(feature: dict, dx: float, dy: float) -> dict:
    """Trasla la geometria di una feature GeoJSON di (dx, dy) (coordinate locali -> globali)."""
    geom = shape(feature["geometry"])
    geom_translated = translate(geom, xoff=dx, yoff=dy)
    new_feature = dict(feature)
    new_feature["geometry"] = mapping(geom_translated)
    return new_feature


def feature_centroid(feature: dict):
    """
    Centroide del BOUNDING BOX del contorno (non centroide geometrico dell'area
    calcolato da shapely).

    Motivo: in caso di nuclei con contorno concavo o distorto (celle sovrapposte,
    split imperfetti tra patch adiacenti) il centroide "vero" (geom.centroid) puo'
    spostarsi rispetto al centro geometrico stabile che si ottiene dal bounding box.
    Usiamo lo stesso criterio qui — per il filtro dei nuclei nel padding e per la
    deduplica nelle zone di overlap — e nello script di valutazione
    (pantumor_inference.py), dove i nuclei predetti vengono confrontati contro un
    GT che e' esso stesso un bounding box. Definire il centroide allo stesso modo
    in tutte le fasi della pipeline evita disallineamenti sistematici.
    """
    geom = shape(feature["geometry"])
    minx, miny, maxx, maxy = geom.bounds
    return (minx + maxx) / 2.0, (miny + maxy) / 2.0


def merge_patches_to_global(patch_results, pad_bottom: int, pad_right: int,
                             original_h: int, original_w: int,
                             overlap: int, dedup_dist_px: float = 8.0):
    """
    patch_results: lista di tuple (geojson_path, y0, x0) in coordinate dell'immagine paddata.

    Step:
      0) Esplode ogni Feature-per-classe nei suoi singoli nuclei (vedi explode_features_per_nucleus).
      a) Traduce ogni nucleo in coordinate globali (= coordinate dell'immagine paddata,
         che condivide l'origine (0,0) con l'originale perché il pad è solo bottom/right).
      b) Rimuove nuclei il cui centroide (bbox-based, vedi feature_centroid) cade
         nell'area di padding (x >= original_w o y >= original_h).
      c) Deduplica nuclei vicini tra patch diversi (stesso nucleo visto da overlap):
         tiene quello con area poligono maggiore.
    """
    all_features = []

    for geojson_path, y0, x0 in patch_results:
        if geojson_path is None:
            continue
        raw_features = load_geojson_features(geojson_path)
        per_nucleus_features = explode_features_per_nucleus(raw_features)
        for feat in per_nucleus_features:
            translated = translate_feature_geometry(feat, dx=x0, dy=y0)
            all_features.append(translated)

    print(f"[MERGE] Totale nuclei prima di filtro/dedup: {len(all_features)}")

    # b) rimozione nuclei nel padding (centroide bbox fuori dall'area originale)
    in_bounds = []
    for feat in all_features:
        cx, cy = feature_centroid(feat)
        if 0 <= cx < original_w and 0 <= cy < original_h:
            in_bounds.append(feat)
    print(f"[MERGE] Dopo rimozione nuclei nel padding: {len(in_bounds)}")

    # c) deduplica nuclei vicini (overlap tra patch)
    deduped = deduplicate_features(in_bounds, dedup_dist_px=dedup_dist_px)
    print(f"[MERGE] Dopo deduplica overlap: {len(deduped)}")

    return deduped


def deduplicate_features(features, dedup_dist_px: float = 8.0):
    """
    Usa una cKDTree (scipy) sui centroidi bbox-based (vedi feature_centroid) per
    trovare nuclei duplicati (stesso nucleo predetto da due patch diversi nella
    zona di overlap). Due nuclei sono considerati lo stesso se i centroidi
    distano meno di 'dedup_dist_px'.

    Tra i duplicati si tiene quello con area maggiore del poligono (di solito più
    completo/centrale nel suo patch di origine), in assenza di altri criteri affidabili.

    Nota: si usa cKDTree invece di shapely.STRtree perché il comportamento di
    STRtree.query() e' cambiato tra Shapely <2.0 (ritorna geometrie) e >=2.0
    (ritorna indici), causando errori se l'ambiente ha una versione diversa
    da quella assunta in fase di sviluppo. cKDTree ha un'API stabile.
    """
    if not features:
        return []

    from scipy.spatial import cKDTree

    geoms = [shape(feat["geometry"]) for feat in features]
    centroids_xy = np.array([
        [(g.bounds[0] + g.bounds[2]) / 2.0, (g.bounds[1] + g.bounds[3]) / 2.0]
        for g in geoms
    ])

    tree = cKDTree(centroids_xy)
    # per ogni punto, trova tutti i vicini entro dedup_dist_px (incluso se stesso)
    pairs = tree.query_pairs(r=dedup_dist_px)  # set di tuple (i, j) con i < j

    # costruisco gruppi di duplicati via union-find semplice
    parent = list(range(len(features)))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j in pairs:
        union(i, j)

    groups = {}
    for idx in range(len(features)):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    keep_mask = [True] * len(features)
    areas = [g.area for g in geoms]

    for group in groups.values():
        if len(group) <= 1:
            continue
        best_idx = max(group, key=lambda idx: areas[idx])
        for idx in group:
            if idx != best_idx:
                keep_mask[idx] = False

    return [feat for feat, keep in zip(features, keep_mask) if keep]


# --------------------------------------------------------------------------------------
# 5. Metriche: precision / recall / F1 di detection + classification report vs GT csv
# --------------------------------------------------------------------------------------

def load_csv_gt(path: Path, scale_x: float = 1.0, scale_y: float = 1.0):
    """
    Legge un CSV con colonne: Cell Type, XMin, YMin, XMax, YMax.
    Calcola il centroide del bbox GT (già coerente con il criterio bbox-centroide
    usato per i nuclei predetti). scale_x/scale_y servono solo se il CSV è in uno
    spazio di coordinate diverso da quello del GeoJSON (di norma qui non serve,
    perché il merge riporta tutto nello spazio dell'immagine originale).
    """
    nuclei = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_raw = row.get("Cell Type", "")
            cls_int = CSV_NAME_TO_CLASS.get(name_raw.lower().strip(), 0)
            if cls_int == 0:
                continue

            xmin, ymin = float(row["XMin"]), float(row["YMin"])
            xmax, ymax = float(row["XMax"]), float(row["YMax"])
            cx = (xmin + xmax) / 2.0 * scale_x
            cy = (ymin + ymax) / 2.0 * scale_y

            nuclei.append({"centroid": (cx, cy), "class": cls_int})

    return nuclei


def merged_features_to_nuclei(features):
    """
    Converte le feature del GeoJSON finale (merged) in {"centroid": (cx, cy), "class": int},
    usando lo stesso centroide bbox-based di feature_centroid() e lo stesso mapping
    classe->intero usato nello script di valutazione.
    """
    nuclei = []
    for feat in features:
        props = feat.get("properties", {})
        clf = props.get("classification", {})
        name_raw = clf.get("name", "") if isinstance(clf, dict) else str(clf)
        cls_int = GEOJSON_NAME_TO_CLASS.get(name_raw.lower().strip(), 0)
        if cls_int == 0:
            continue
        cx, cy = feature_centroid(feat)
        nuclei.append({"centroid": (cx, cy), "class": cls_int})
    return nuclei


def match_nuclei(pred_list, gt_list, dist_thresh):
    """Identico al matching in pantumor_inference.py: KD-Tree sui centroidi bbox-based."""
    if not pred_list or not gt_list:
        return [], [], len(pred_list), len(gt_list)

    gt_pts = np.array([n["centroid"] for n in gt_list])
    pred_pts = np.array([n["centroid"] for n in pred_list])

    tree = cKDTree(gt_pts)
    dists, idxs = tree.query(pred_pts, k=1)

    matched_gt = set()
    y_true, y_pred = [], []
    n_fp = 0

    for i, (d, j) in enumerate(zip(dists, idxs)):
        if d <= dist_thresh and j not in matched_gt:
            matched_gt.add(j)
            y_true.append(gt_list[j]["class"])
            y_pred.append(pred_list[i]["class"])
        else:
            n_fp += 1

    n_fn = len(gt_list) - len(matched_gt)
    return y_true, y_pred, n_fp, n_fn


def save_confusion_matrix(cm, class_names, out_path):
    if not HAS_MPL:
        print("  [WARN] matplotlib assente, confusion matrix non salvata.")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predetto", ylabel="Ground Truth",
           title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix -> {out_path}")


def evaluate_against_gt(merged_features, gt_csv_path: Path, dist_thresh: float,
                         scale_x: float, scale_y: float, out_dir: Path):
    """
    Confronta i nuclei del GeoJSON finale (merged) con il GT csv e stampa/salva:
    precision/recall/F1 di detection + classification report + confusion matrix.
    Ritorna (y_true, y_pred, n_fp, n_fn) per poter aggregare su più immagini in batch.
    """
    if not HAS_SKLEARN:
        print("  [WARN] scikit-learn assente: pip install scikit-learn --break-system-packages. "
              "Metriche saltate.")
        return [], [], 0, 0

    pred_nuclei = merged_features_to_nuclei(merged_features)
    gt_nuclei = load_csv_gt(gt_csv_path, scale_x=scale_x, scale_y=scale_y)
    y_true, y_pred, n_fp, n_fn = match_nuclei(pred_nuclei, gt_nuclei, dist_thresh)

    n_tp = len(y_true)
    det_p = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    det_r = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    det_f1 = 2 * det_p * det_r / (det_p + det_r) if (det_p + det_r) > 0 else 0.0

    print(f"  [METRICS] pred={len(pred_nuclei)}  gt={len(gt_nuclei)}  "
          f"TP={n_tp}  FP={n_fp}  FN={n_fn}  "
          f"P={det_p:.4f}  R={det_r:.4f}  F1={det_f1:.4f}")

    if not y_true:
        print("  [WARN] Nessun nucleo matchato: controlla dist_thresh/scale_x/scale_y.")
        return y_true, y_pred, n_fp, n_fn

    out_dir.mkdir(parents=True, exist_ok=True)
    present = sorted(set(y_true) | set(y_pred))
    names = [GT_CLASS_NAMES.get(c, str(c)) for c in present]
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=present,
                                    target_names=names, digits=4, zero_division=0)

    lines = [
        "=" * 65,
        "  NuLite (patch_infer_merge) -> GT csv  |  Evaluation Report",
        f"  dist_thresh = {dist_thresh} px",
        "=" * 65,
        "",
        "-- Detection (tutti i nuclei) --------------------------------",
        f"   TP = {n_tp}   FP = {n_fp}   FN = {n_fn}",
        f"   Precision : {det_p:.4f}",
        f"   Recall    : {det_r:.4f}",
        f"   F1        : {det_f1:.4f}",
        "",
        f"-- Classificazione (sui {n_tp} TP matchati) --------------",
        f"   Accuracy  : {acc:.4f}",
        "",
        report,
    ]
    text = "\n".join(lines)
    print(text)

    rpt_path = out_dir / "classification_report.txt"
    with open(rpt_path, "w") as f:
        f.write(text)
    print(f"  Report -> {rpt_path}")

    cm = confusion_matrix(y_true, y_pred, labels=present)
    save_confusion_matrix(cm, names, out_dir / "confusion_matrix.png")

    return y_true, y_pred, n_fp, n_fn


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def process_single_image(image_path: Path, output_dir: Path, inference_script: Path,
                          model_path: Path, gpu: int, magnification: int,
                          patch_size: int, overlap: int, dedup_dist_px: float,
                          keep_patches: bool, gt_csv_path: Path = None,
                          dist_thresh: float = 25.0, scale_x: float = 1.0,
                          scale_y: float = 1.0):
    """
    Esegue l'intera pipeline (padding -> patching -> inference -> merge) su una
    singola immagine, salvando il risultato in output_dir/<stem>_merged.geojson.
    Se gt_csv_path è fornito, calcola anche precision/recall/F1 + classification
    report confrontando il GeoJSON finale con il GT, salvati in output_dir/metrics/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = output_dir / "patches"

    stride = patch_size - overlap
    assert stride > 0, "overlap deve essere minore di patch_size"

    print(f"[1/4] Carico immagine: {image_path}")
    img = np.array(Image.open(image_path).convert("RGB"))
    original_h, original_w = img.shape[0], img.shape[1]
    print(f"      Dimensione originale: {original_w}x{original_h}")

    print(f"[2/4] Padding reflect (patch_size={patch_size}, stride={stride})")
    img_padded, pad_bottom, pad_right = pad_image_reflect(img, patch_size, stride)
    padded_h, padded_w = img_padded.shape[0], img_padded.shape[1]
    print(f"      Dimensione paddata: {padded_w}x{padded_h} (pad_bottom={pad_bottom}, pad_right={pad_right})")

    coords = generate_patch_coords(padded_h, padded_w, patch_size, stride)
    print(f"      Numero di patch generati: {len(coords)}")

    patch_info = save_patches(img_padded, coords, patch_size, patches_dir)

    print(f"[3/4] Inference su {len(patch_info)} patch...")
    patch_results = []
    for idx, (patch_path, y0, x0) in enumerate(patch_info):
        print(f"      ({idx + 1}/{len(patch_info)}) {patch_path.name}")
        geojson_path = run_inference_on_patch(
            inference_script=inference_script,
            model_path=model_path,
            image_path=patch_path,
            gpu=gpu,
            magnification=magnification,
        )
        patch_results.append((geojson_path, y0, x0))

    print(f"[4/4] Merge dei risultati...")
    merged_features = merge_patches_to_global(
        patch_results,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
        original_h=original_h,
        original_w=original_w,
        overlap=overlap,
        dedup_dist_px=dedup_dist_px,
    )

    output_geojson = {
        "type": "FeatureCollection",
        "features": merged_features,
    }
    out_path = output_dir / f"{image_path.stem}_merged.geojson"
    with open(out_path, "w") as f:
        json.dump(output_geojson, f)

    print(f"\nFatto. GeoJSON finale: {out_path}")
    print(f"Totale nuclei finali: {len(merged_features)}")

    if not keep_patches:
        shutil.rmtree(patches_dir, ignore_errors=True)
        print("Cartella patch intermedi rimossa (usa --keep_patches per conservarla).")

    y_true, y_pred, n_fp, n_fn = [], [], 0, 0
    if gt_csv_path is not None:
        print(f"\n[METRICS] Valutazione contro GT: {gt_csv_path}")
        y_true, y_pred, n_fp, n_fn = evaluate_against_gt(
            merged_features, gt_csv_path, dist_thresh, scale_x, scale_y,
            out_dir=output_dir / "metrics",
        )

    return out_path, len(merged_features), y_true, y_pred, n_fp, n_fn


def main():
    parser = argparse.ArgumentParser(description="Patching + inference NuLite + merge per immagini > 1024x1024")
    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument("--image", type=str, help="Path a una singola immagine (es. 2220x2220)")
    image_group.add_argument("--image_dir", type=str,
                              help="Cartella contenente più immagini da processare in batch "
                                   f"(estensioni supportate: {sorted(IMAGE_EXTENSIONS)})")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Cartella di output. In modalità batch, per ogni immagine <nome>.tif "
                              "viene creata una sottocartella output_dir/<nome>/")
    parser.add_argument("--inference_script", type=str, required=True, help="Path a inference_patch.py")
    parser.add_argument("--model", type=str, required=True, help="Path al checkpoint del modello")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--magnification", type=int, default=20)
    parser.add_argument("--patch_size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--dedup_dist_px", type=float, default=8.0,
                         help="Distanza massima tra centroidi per considerare due nuclei duplicati")
    parser.add_argument("--keep_patches", action="store_true",
                         help="Se settato, non elimina la cartella dei patch intermedi a fine run")
    parser.add_argument("--skip_existing", action="store_true",
                         help="In modalità batch, salta le immagini per cui esiste già <stem>_merged.geojson")
    parser.add_argument("--gt", type=str, default=None,
                         help="Singolo file .csv GT (solo modalità --image). Se fornito, calcola "
                              "precision/recall/F1 + classification report a fine merge.")
    parser.add_argument("--gt_dir", type=str, default=None,
                         help="Cartella con i .csv GT (solo modalità --image_dir), stesso stem "
                              "delle immagini. Se un'immagine non ha un GT corrispondente, viene "
                              "comunque processata ma senza metriche.")
    parser.add_argument("--dist_thresh", type=float, default=25.0,
                         help="Soglia distanza (px) tra centroidi bbox per il matching pred/GT (default 25)")
    parser.add_argument("--scale_x", type=float, default=1.0,
                         help="Fattore di scala X da applicare alle coordinate del GT csv (default 1.0, "
                              "nessuna riscalatura: usalo solo se il GT non è nello spazio dell'immagine originale)")
    parser.add_argument("--scale_y", type=float, default=1.0,
                         help="Fattore di scala Y da applicare alle coordinate del GT csv (default 1.0)")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    inference_script = Path(args.inference_script)
    model_path = Path(args.model)

    common_kwargs = dict(
        inference_script=inference_script,
        model_path=model_path,
        gpu=args.gpu,
        magnification=args.magnification,
        patch_size=args.patch_size,
        overlap=args.overlap,
        dedup_dist_px=args.dedup_dist_px,
        keep_patches=args.keep_patches,
        dist_thresh=args.dist_thresh,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )

    if args.image:
        # Modalità singola immagine (comportamento originale)
        image_path = Path(args.image)
        output_root.mkdir(parents=True, exist_ok=True)
        gt_csv_path = Path(args.gt) if args.gt else None
        process_single_image(image_path=image_path, output_dir=output_root,
                              gt_csv_path=gt_csv_path, **common_kwargs)
        return

    # Modalità batch su cartella
    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"[BATCH] Trovate {len(image_paths)} immagini in {image_dir}")

    results_summary = []
    all_y_true, all_y_pred = [], []
    total_fp = total_fn = 0

    for idx, image_path in enumerate(image_paths):
        # La cartella di output per ogni immagine prende il nome dallo stem del file
        # (es. GC_01.tif -> output_dir/GC_01/)
        image_output_dir = output_root / image_path.stem
        expected_out = image_output_dir / f"{image_path.stem}_merged.geojson"

        if args.skip_existing and expected_out.exists():
            print(f"\n[BATCH] ({idx + 1}/{len(image_paths)}) Salto {image_path.name}: "
                  f"output già esistente ({expected_out})")
            continue

        print(f"\n{'=' * 80}")
        print(f"[BATCH] ({idx + 1}/{len(image_paths)}) Processo: {image_path.name}")
        print(f"{'=' * 80}")

        gt_csv_path = None
        if gt_dir is not None:
            candidate = gt_dir / f"{image_path.stem}.csv"
            if candidate.exists():
                gt_csv_path = candidate
            else:
                print(f"  [WARN] Nessun GT trovato per {image_path.stem} in {gt_dir}, "
                      f"salto le metriche per questa immagine.")

        try:
            out_path, n_nuclei, y_true, y_pred, n_fp, n_fn = process_single_image(
                image_path=image_path, output_dir=image_output_dir,
                gt_csv_path=gt_csv_path, **common_kwargs
            )
            results_summary.append((image_path.name, n_nuclei, "OK"))
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            total_fp += n_fp
            total_fn += n_fn
        except Exception as e:
            print(f"[BATCH][ERRORE] Fallito su {image_path.name}: {e}")
            results_summary.append((image_path.name, None, f"ERRORE: {e}"))
            continue

    print(f"\n{'=' * 80}")
    print("[BATCH] Riepilogo finale:")
    print(f"{'=' * 80}")
    for name, n_nuclei, status in results_summary:
        if status == "OK":
            print(f"  {name}: {n_nuclei} nuclei -> OK")
        else:
            print(f"  {name}: {status}")

    # Metriche aggregate su tutto il batch (solo se almeno un'immagine aveva un GT)
    if gt_dir is not None and HAS_SKLEARN and all_y_true:
        n_tp = len(all_y_true)
        det_p = n_tp / (n_tp + total_fp) if (n_tp + total_fp) > 0 else 0.0
        det_r = n_tp / (n_tp + total_fn) if (n_tp + total_fn) > 0 else 0.0
        det_f1 = 2 * det_p * det_r / (det_p + det_r) if (det_p + det_r) > 0 else 0.0
        present = sorted(set(all_y_true) | set(all_y_pred))
        names = [GT_CLASS_NAMES.get(c, str(c)) for c in present]
        acc = accuracy_score(all_y_true, all_y_pred)
        report = classification_report(all_y_true, all_y_pred, labels=present,
                                        target_names=names, digits=4, zero_division=0)

        agg_lines = [
            "",
            "=" * 65,
            "  [BATCH] Metriche AGGREGATE su tutte le immagini con GT",
            f"  dist_thresh = {args.dist_thresh} px",
            "=" * 65,
            f"   TP = {n_tp}   FP = {total_fp}   FN = {total_fn}",
            f"   Precision : {det_p:.4f}",
            f"   Recall    : {det_r:.4f}",
            f"   F1        : {det_f1:.4f}",
            f"   Accuracy (sui TP matchati) : {acc:.4f}",
            "",
            report,
        ]
        agg_text = "\n".join(agg_lines)
        print(agg_text)

        agg_out_dir = output_root / "metrics_aggregate"
        agg_out_dir.mkdir(parents=True, exist_ok=True)
        with open(agg_out_dir / "classification_report.txt", "w") as f:
            f.write(agg_text)
        cm = confusion_matrix(all_y_true, all_y_pred, labels=present)
        save_confusion_matrix(cm, names, agg_out_dir / "confusion_matrix.png")
        print(f"  Report aggregato -> {agg_out_dir}")


if __name__ == "__main__":
    main()
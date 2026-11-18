import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
except ImportError:
    print("[ERROR] Installa scikit-learn:  pip install scikit-learn")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── Costanti ─────────────────────────────────────────────────────────────────

GT_CLASS_NAMES = {1: "Neoplastic", 2: "Inflammatory", 3: "Other"}

# Mapping classe del geojson (output modello) → intero comune
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
}

# Mapping "Cell Type" del CSV (ground truth) → stesso intero comune
# Adatta qui se il tuo CSV usa nomi diversi.
CSV_NAME_TO_CLASS = {
    "tumor cell": 1,            # → Neoplastic
    "cd3+ immune cell": 2,       # → Inflammatory
    "other cell": 3,             # → Other
}


# ── Lettura GeoJSON (predizioni modello, coordinate nello spazio target, es. 1024x1024) ──

def centroid_from_polygon(coords):
    pts = np.array(coords[0])
    return pts[:, 0].mean(), pts[:, 1].mean()


def load_geojson(path: Path):
    with open(path) as f:
        data = json.load(f)

    features = data if isinstance(data, list) else data.get("features", [])
    nuclei = []

    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})

        clf = props.get("classification", {})
        name_raw = clf.get("name", "") if isinstance(clf, dict) else str(clf)
        cls_int = GEOJSON_NAME_TO_CLASS.get(name_raw.lower().strip(), 0)
        if cls_int == 0:
            continue

        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        if gtype == "Polygon":
            cx, cy = centroid_from_polygon(coords)
            nuclei.append({"centroid": (cx, cy), "class": cls_int})

        elif gtype == "MultiPolygon":
            for poly_coords in coords:
                while len(poly_coords) == 1 and isinstance(poly_coords[0], list):
                    poly_coords = poly_coords[0]
                pts = np.array(poly_coords)
                if pts.ndim == 2 and pts.shape[1] == 2:
                    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                    nuclei.append({"centroid": (cx, cy), "class": cls_int})

        elif gtype == "Point":
            nuclei.append({"centroid": (coords[0], coords[1]), "class": cls_int})

        elif gtype == "MultiPoint":
            for c in coords:
                nuclei.append({"centroid": (c[0], c[1]), "class": cls_int})

    return nuclei


# ── Lettura CSV (ground truth, bbox in coordinate immagine ORIGINALE) ────────

def load_csv_gt(path: Path, scale_x: float = 1.0, scale_y: float = 1.0):
    """
    Legge un CSV con colonne: Cell Type, XMin, YMin, XMax, YMax.
    Calcola il centroide del bbox e lo riscala (scale_x, scale_y) per portarlo
    nello stesso sistema di coordinate del geojson predetto (es. 1024x1024).
    Restituisce lista di dict {"centroid": (cx, cy), "class": int}.
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


def read_image_size(img_path: Path):
    """Legge solo l'header dell'immagine (via PIL, senza decodificare i pixel),
    funziona con qualunque formato supportato da Pillow (tif, png, jpg, ...)."""
    if not HAS_PIL:
        raise RuntimeError("Pillow non disponibile: pip install Pillow")
    with Image.open(img_path) as im:
        return im.size  # (w, h)


def get_scale_factors(orig_path: Path, converted_path: Path = None, target_size: int = 1024):
    """
    Calcola i fattori di scala (scale_x, scale_y) da applicare alle coordinate
    del CSV (spazio immagine originale) per portarle nello spazio del geojson
    (spazio immagine convertita, es. il png dato in input al modello).

    - Se converted_path è fornito, legge la sua dimensione REALE (più robusto,
      copre anche resize non perfettamente quadrati).
    - Altrimenti assume che la convertita sia target_size x target_size.
    """
    orig_w, orig_h = read_image_size(orig_path)
    if converted_path is not None:
        conv_w, conv_h = read_image_size(converted_path)
    else:
        conv_w, conv_h = target_size, target_size
    return conv_w / orig_w, conv_h / orig_h


# ── Matching centroidi (identico al tuo eval_consep.py) ──────────────────────

def match_nuclei(pred_list, gt_list, dist_thresh):
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


# ── Confusion matrix ──────────────────────────────────────────────────────────

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
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix → {out_path}")


# ── Valutazione su una coppia geojson/csv ─────────────────────────────────────

def evaluate_one(geojson_path, csv_path, dist_thresh, scale_x, scale_y):
    pred_nuclei = load_geojson(geojson_path)
    gt_nuclei = load_csv_gt(csv_path, scale_x=scale_x, scale_y=scale_y)
    y_true, y_pred, n_fp, n_fn = match_nuclei(pred_nuclei, gt_nuclei, dist_thresh)
    return y_true, y_pred, n_fp, n_fn, len(pred_nuclei), len(gt_nuclei)


# ── Report testuale ───────────────────────────────────────────────────────────

def print_and_save_report(y_true, y_pred, total_fp, total_fn, out_dir, dist_thresh):
    out_dir.mkdir(parents=True, exist_ok=True)

    present = sorted(set(y_true) | set(y_pred))
    names = [GT_CLASS_NAMES.get(c, str(c)) for c in present]

    total_tp = len(y_true)
    det_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    det_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    det_f1 = 2 * det_p * det_r / (det_p + det_r) if (det_p + det_r) > 0 else 0

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=present,
                                    target_names=names, digits=4, zero_division=0)

    lines = [
        "=" * 65,
        "  NuLite -> IHC (CSV bbox GT)  |  Evaluation Report",
        f"  dist_thresh = {dist_thresh} px",
        "=" * 65,
        "",
        "-- Detection (tutti i nuclei) --------------------------------",
        f"   TP = {total_tp}   FP = {total_fp}   FN = {total_fn}",
        f"   Precision : {det_p:.4f}",
        f"   Recall    : {det_r:.4f}",
        f"   F1        : {det_f1:.4f}",
        "",
        f"-- Classificazione (sui {total_tp} TP matchati) --------------",
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Metriche NuLite (geojson) vs GT IHC (.csv bbox)"
    )

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--geojson", type=Path, help="Singolo file .geojson")
    grp.add_argument("--geojson_dir", type=Path, help="Cartella con .geojson (uno per immagine)")

    csv_grp = parser.add_mutually_exclusive_group(required=True)
    csv_grp.add_argument("--csv", type=Path, help="Singolo file .csv GT")
    csv_grp.add_argument("--csv_dir", type=Path, help="Cartella con i .csv GT (stem deve coincidere col geojson)")

    parser.add_argument("--orig", type=Path, default=None,
                         help="Singolo file immagine ORIGINALE (.tif), per leggere la dimensione e calcolare lo scale factor")
    parser.add_argument("--orig_dir", type=Path, default=None,
                         help="Cartella con le immagini ORIGINALI (.tif), stesso stem dei csv")
    parser.add_argument("--converted", type=Path, default=None,
                         help="Singolo file immagine CONVERTITA (.png) data in input al modello, per leggere la dimensione reale")
    parser.add_argument("--converted_dir", type=Path, default=None,
                         help="Cartella con le immagini CONVERTITE (.png), stesso stem dei csv. Se omessa, si assume target_size x target_size")
    parser.add_argument("--target_size", type=int, default=1024,
                         help="Dimensione (lato) assunta per le immagini convertite SOLO se non si passa --converted/--converted_dir (default 1024)")
    parser.add_argument("--scale_x", type=float, default=None,
                         help="Fattore di scala X manuale, se non si passa --tif/--tif_dir (default 1.0, nessuna riscalatura)")
    parser.add_argument("--scale_y", type=float, default=None,
                         help="Fattore di scala Y manuale, se non si passa --tif/--tif_dir (default 1.0, nessuna riscalatura)")
    parser.add_argument("--output", type=Path, default=Path("metrics_ihc"),
                         help="Cartella di output (default: ./metrics_ihc)")
    parser.add_argument("--dist_thresh", type=float, default=25.0,
                         help="Soglia distanza centroide in px, nello spazio target_size (default 25)")
    args = parser.parse_args()

    # ── Raccogli coppie (geojson, csv, orig_img|None, converted_img|None) ────
    pairs = []

    IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

    def find_with_stem(directory, stem):
        if directory is None:
            return None
        for ext in IMG_EXTS:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    if args.geojson and args.csv:
        pairs = [(args.geojson, args.csv, args.orig, args.converted)]

    elif args.geojson_dir and args.csv_dir:
        for csv_path in sorted(args.csv_dir.glob("*.csv")):
            stem = csv_path.stem
            gj = args.geojson_dir / f"{stem}.geojson"
            if not gj.exists():
                gj = args.geojson_dir / stem / "cell_detection" / "cells.geojson"
            if not gj.exists():
                print(f"[SKIP] {stem}: nessun .geojson trovato")
                continue

            orig_path = find_with_stem(args.orig_dir, stem)
            if args.orig_dir is not None and orig_path is None:
                print(f"[SKIP] {stem}: immagine originale non trovata in --orig_dir")
                continue

            converted_path = find_with_stem(args.converted_dir, stem)
            if args.converted_dir is not None and converted_path is None:
                print(f"[SKIP] {stem}: immagine convertita non trovata in --converted_dir")
                continue

            pairs.append((gj, csv_path, orig_path, converted_path))

    else:
        print("[ERROR] Combinazione di argomenti non valida.")
        parser.print_help()
        sys.exit(1)

    if not pairs:
        print("[ERROR] Nessuna coppia geojson/csv trovata.")
        sys.exit(1)

    print(f"\nCoppie trovate: {len(pairs)}  |  dist_thresh={args.dist_thresh}px  |  target_size={args.target_size}\n")

    # ── Valutazione ───────────────────────────────────────────────────────────
    all_y_true, all_y_pred = [], []
    total_fp = total_fn = 0
    per_img_rows = []

    for gj_path, csv_path, orig_path, converted_path in pairs:
        print(f"  {csv_path.stem} ...", end=" ", flush=True)

        if args.scale_x is not None or args.scale_y is not None:
            # Override manuale: ignora orig/converted, usa i valori passati (default 1.0 se uno dei due manca)
            scale_x = args.scale_x if args.scale_x is not None else 1.0
            scale_y = args.scale_y if args.scale_y is not None else 1.0
        elif orig_path is not None:
            scale_x, scale_y = get_scale_factors(orig_path, converted_path, args.target_size)
        else:
            # Nessuna immagine e nessun override manuale → nessuna riscalatura
            scale_x, scale_y = 1.0, 1.0

        y_true, y_pred, n_fp, n_fn, n_pred, n_gt = evaluate_one(
            gj_path, csv_path, args.dist_thresh, scale_x, scale_y
        )
        n_tp = len(y_true)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        total_fp += n_fp
        total_fn += n_fn

        img_acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        print(f"pred={n_pred}  gt={n_gt}  TP={n_tp}  FP={n_fp}  FN={n_fn}  acc={img_acc:.3f}  (scale={scale_x:.4f},{scale_y:.4f})")

        per_img_rows.append({
            "image": csv_path.stem,
            "n_pred": n_pred, "n_gt": n_gt,
            "TP": n_tp, "FP": n_fp, "FN": n_fn,
            "accuracy": round(img_acc, 4),
        })

    if not all_y_true:
        print("\n[ERROR] Nessun nucleo matchato. Controlla dist_thresh, scale_x/scale_y e i path.")
        sys.exit(1)

    print()
    print_and_save_report(all_y_true, all_y_pred, total_fp, total_fn,
                           args.output, args.dist_thresh)

    csv_out_path = args.output / "per_image.csv"
    with open(csv_out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "n_pred", "n_gt", "TP", "FP", "FN", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(per_img_rows)
    print(f"  Per-image CSV -> {csv_out_path}")


if __name__ == "__main__":
    main()
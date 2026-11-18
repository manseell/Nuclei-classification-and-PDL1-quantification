"""
eval_consep.py
──────────────────────────────────────────────────────────────────────────────
Calcola accuracy, recall e F1 (totale e per classe) confrontando:
  - Predizioni NuLite  → file .geojson (MultiPolygon + classification.name)
  - Ground Truth       → file .npy     (inst_map + type_map, 1000×1000)

Classi GT nel .npy (dopo remap CoNSeP):
  0 = Background (ignorato)
  1 = Other
  2 = Inflammatory
  3 = Epithelial

Mapping stringa geojson → intero GT:
  "Other"          → 3
  "Inflammatory"   → 2
  "Neoplastic"     → 1
  (qualsiasi altro → 0, scartato)

Strategia di matching:
  Per ogni nucleo predetto (centroide = media dei vertici del poligono)
  si cerca il nucleo GT più vicino per distanza euclidea tra centroidi.
  Se distanza ≤ dist_thresh  →  True Positive  (poi confronta la classe)
  Altrimenti                 →  False Positive
  Nuclei GT senza match      →  False Negative

Uso:
  # Singola immagine
  python eval_consep.py \\
      --geojson  path/to/cells.geojson \\
      --npy      path/to/test_1.npy \\
      --output   metrics/

  # Batch (una cartella di geojson + una di npy, stessi stem)
  python eval_consep.py \\
      --geojson_dir  path/to/geojsons/ \\
      --npy_dir      path/to/npy/ \\
      --output       metrics/

  Opzionale: --dist_thresh 12   (default 12 px, scala 1000×1000)
             --npy_size 1024    (se hai ridimensionato i .npy a 1024×1024)
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from collections import defaultdict

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


# ── Costanti ─────────────────────────────────────────────────────────────────

GT_CLASS_NAMES = {1: "Neoplastic", 2: "Inflammatory", 3: "Other"}

# Mapping stringa geojson → intero GT

NAME_TO_GT = {
    "other":          3,   # ✓
    "inflammatory":   2,   # ✓
    "neoplastic":     1,   # ✓
    # Aggiunti per sicurezza nonostante non ci siano nel modello
    "epithelial":     1,   # trattalo come Neoplastic
    "connective":     3,   # trattalo come Other
    "dead":           3,   # trattalo come Other
    "spindle_shaped": 3,
    "spindle-shaped": 3,
}
'''
NAME_TO_GT = {
    "other":          3,   # ✓
    "inflammatory":   2,   # ✓
    "neoplastic":     1,   # ✓
    # Aggiunti per sicurezza nonostante non ci siano nel modello
    "dysplastic_epithelial":     1,   # trattalo come Neoplastic
    "fibroblast":     3,   # trattalo come Other
    "muscle":           3,   # trattalo come Other
    "endothelial": 3,
}'''


# ── Lettura GeoJSON ───────────────────────────────────────────────────────────

def centroid_from_polygon(coords):
    """Centroide come media dei vertici del primo anello del poligono."""
    pts = np.array(coords[0])   # lista di [x, y]
    return pts[:, 0].mean(), pts[:, 1].mean()


def load_geojson(path: Path):
    """
    Restituisce lista di dict {"centroid": (cx, cy), "class": int}
    per ogni nucleo nel geojson (ignora Background/classe 0).
    """
    with open(path) as f:
        data = json.load(f)

    features = data if isinstance(data, list) else data.get("features", [])
    nuclei = []

    for feat in features:
        geom  = feat.get("geometry", {})
        props = feat.get("properties", {})

        # ---- classe ----
        clf      = props.get("classification", {})
        name_raw = clf.get("name", "") if isinstance(clf, dict) else str(clf)
        cls_int  = NAME_TO_GT.get(name_raw.lower().strip(), 0)
        if cls_int == 0:
            continue   # Background o classe sconosciuta → salta

        # ---- centroide ----
        gtype  = geom.get("type", "")
        coords = geom.get("coordinates", [])

        
        # Gestisce la detection del centroide in base alla tipologia di classe
        
        if gtype == "Polygon":
            # Calcolo il centroide dal poligono
            cx, cy = centroid_from_polygon(coords)
            nuclei.append({"centroid": (cx, cy), "class": cls_int})

        elif gtype == "MultiPolygon":
            # Considero separatamente ogni sotto-poligono (nucleo)
            for poly_coords in coords:
                # Se l'array è troppo annidato, scendiamo fino a trovare i punti [x, y]
                while len(poly_coords) == 1 and isinstance(poly_coords[0], list):
                    poly_coords = poly_coords[0]
                
                pts = np.array(poly_coords)
                if pts.ndim == 2 and pts.shape[1] == 2:
                    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                    nuclei.append({"centroid": (cx, cy), "class": cls_int})

        elif gtype == "Point":
            # Aggiungo il punto come centroide
            nuclei.append({"centroid": (coords[0], coords[1]), "class": cls_int})

        elif gtype == "MultiPoint":
            # Considero separatamente ogni punto
            for c in coords:
                nuclei.append({"centroid": (c[0], c[1]), "class": cls_int})

    return nuclei


# ── Lettura NPY (GT) ──────────────────────────────────────────────────────────

def load_npy_gt(path: Path, target_size: int = None):
    """
    Carica inst_map e type_map dal .npy.
    Se target_size è specificato, ridimensiona le mappe (INTER_NEAREST).
    Restituisce lista di dict {"centroid": (cx, cy), "class": int}.
    """
    data     = np.load(str(path), allow_pickle=True).item()
    inst_map = data["inst_map"].astype(np.int32)
    type_map = data["type_map"].astype(np.int32)

    # Ridimensionamento opzionale
    if target_size is not None:
        orig_H, orig_W = inst_map.shape
        if orig_H != target_size or orig_W != target_size:
            import cv2
            inst_map = cv2.resize(inst_map, (target_size, target_size),
                                  interpolation=cv2.INTER_NEAREST)
            type_map = cv2.resize(type_map, (target_size, target_size),
                                  interpolation=cv2.INTER_NEAREST)

    nuclei = []
    for inst_id in np.unique(inst_map):
        if inst_id == 0:
            continue
        mask = inst_map == inst_id
        ys, xs = np.where(mask)
        cx, cy = xs.mean(), ys.mean()

        # Classe = valore più frequente nella maschera (escluso 0)
        labels = type_map[mask]
        labels = labels[labels > 0]
        if len(labels) == 0:
            continue
        gt_class = int(np.bincount(labels).argmax())
        nuclei.append({"centroid": (cx, cy), "class": gt_class})

    return nuclei


# ── Matching centroidi ────────────────────────────────────────────────────────

def match_nuclei(pred_list, gt_list, dist_thresh):
    """
    Abbina ogni nucleo predetto al GT più vicino entro dist_thresh.
    Restituisce:
      y_true, y_pred  → liste parallele di classi (solo TP)
      n_fp            → falsi positivi (predetti senza match GT)
      n_fn            → falsi negativi (GT senza match predetto)
    """
    if not pred_list or not gt_list:
        return [], [], len(pred_list), len(gt_list)

    gt_pts   = np.array([n["centroid"] for n in gt_list])
    pred_pts = np.array([n["centroid"] for n in pred_list])

    # Viene utilizzato il KD-Tree, cioè una struttura ad albero per trovare il punto più vicino
    tree          = cKDTree(gt_pts)
    # Per ogni punto predetto si interroga l'albero
    dists, idxs   = tree.query(pred_pts, k=1)

    matched_gt    = set()         # E' usato per evitare di ricontrollare più di una volta lo stesso nucleo
    y_true, y_pred = [], []
    n_fp = 0

    for i, (d, j) in enumerate(zip(dists, idxs)):
        # Controllo la distanza tra nucleo predetto e nucleo GT
        # Se la distanza è minore di 12px, allora sono considerati come stesso nucleo
        if d <= dist_thresh and j not in matched_gt:
            matched_gt.add(j)
            y_true.append(gt_list[j]["class"])
            y_pred.append(pred_list[i]["class"])
        # Altrimenti è considerato come false positive
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


# ── Valutazione su una coppia geojson/npy ─────────────────────────────────────

def evaluate_one(geojson_path, npy_path, dist_thresh, npy_size):
    pred_nuclei = load_geojson(geojson_path)
    gt_nuclei   = load_npy_gt(npy_path, target_size=npy_size)
    y_true, y_pred, n_fp, n_fn = match_nuclei(pred_nuclei, gt_nuclei, dist_thresh)
    return y_true, y_pred, n_fp, n_fn, len(pred_nuclei), len(gt_nuclei)


# ── Report testuale ───────────────────────────────────────────────────────────

def print_and_save_report(y_true, y_pred, total_fp, total_fn, out_dir, dist_thresh):
    out_dir.mkdir(parents=True, exist_ok=True)

    present = sorted(set(y_true) | set(y_pred))
    names   = [GT_CLASS_NAMES.get(c, str(c)) for c in present]

    total_tp = len(y_true)
    det_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    det_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    det_f1 = 2*det_p*det_r / (det_p+det_r) if (det_p+det_r) > 0 else 0

    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=present,
                                   target_names=names, digits=4, zero_division=0)

    lines = [
        "═" * 65,
        "  NuLite → CoNSeP  |  Evaluation Report",
        f"  dist_thresh = {dist_thresh} px",
        "═" * 65,
        "",
        "── Detection (tutti i nuclei) ──────────────────────────────",
        f"   TP = {total_tp}   FP = {total_fp}   FN = {total_fn}",
        f"   Precision : {det_p:.4f}",
        f"   Recall    : {det_r:.4f}",
        f"   F1        : {det_f1:.4f}",
        "",
        f"── Classificazione (sui {total_tp} TP matchati) ─────────────",
        f"   Accuracy  : {acc:.4f}",
        "",
        report,
    ]
    text = "\n".join(lines)
    print(text)

    rpt_path = out_dir / "classification_report.txt"
    with open(rpt_path, "w") as f:
        f.write(text)
    print(f"  Report → {rpt_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present)
    save_confusion_matrix(cm, names, out_dir / "confusion_matrix.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Metriche NuLite vs GT CoNSeP (.npy)"
    )

    # Input: singolo o batch
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--geojson",     type=Path,
                     help="Singolo file .geojson")
    grp.add_argument("--geojson_dir", type=Path,
                     help="Cartella con un .geojson per immagine")

    npy_grp = parser.add_mutually_exclusive_group(required=True)
    npy_grp.add_argument("--npy",     type=Path,
                          help="Singolo file .npy GT")
    npy_grp.add_argument("--npy_dir", type=Path,
                          help="Cartella con i .npy GT (stem deve coincidere col geojson)")

    parser.add_argument("--output",      type=Path, default=Path("metrics"),
                        help="Cartella di output (default: ./metrics)")
    parser.add_argument("--dist_thresh", type=float, default=25.0,
                        help="Soglia distanza centroide in px (default 12)")
    parser.add_argument("--npy_size",    type=int, default=None,
                        help="Se i .npy vanno ridimensionati (es. 1024), specifica qui")
    args = parser.parse_args()

    # ── Raccogli coppie (geojson, npy) ────────────────────────────────────────
    pairs = []

    if args.geojson and args.npy:
        pairs = [(args.geojson, args.npy)]

    elif args.geojson_dir and args.npy_dir:
        for gj in sorted(args.geojson_dir.glob("*.geojson")):
            npy = args.npy_dir / (gj.stem + ".npy")
            if not npy.exists():
                print(f"[SKIP] {gj.stem}: .npy non trovato")
                continue
            pairs.append((gj, npy))

    elif args.geojson and args.npy_dir:
        # geojson singolo → cerca il .npy con lo stesso stem
        stem = args.geojson.stem
        npy  = args.npy_dir / (stem + ".npy")
        if not npy.exists():
            print(f"[ERROR] {npy} non trovato")
            sys.exit(1)
        pairs = [(args.geojson, npy)]

    else:
        print("[ERROR] Combinazione di argomenti non valida.")
        parser.print_help()
        sys.exit(1)

    if not pairs:
        print("[ERROR] Nessuna coppia geojson/npy trovata.")
        sys.exit(1)

    print(f"\nCoppie trovate: {len(pairs)}  |  dist_thresh={args.dist_thresh}px\n")

    # ── Inferenza ─────────────────────────────────────────────────────────────
    all_y_true, all_y_pred = [], []
    total_fp = total_fn = 0
    per_img_rows = []

    for gj_path, npy_path in pairs:
        print(f"  {gj_path.stem} ...", end=" ", flush=True)
        y_true, y_pred, n_fp, n_fn, n_pred, n_gt = evaluate_one(
            gj_path, npy_path, args.dist_thresh, args.npy_size
        )
        n_tp = len(y_true)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        total_fp += n_fp
        total_fn += n_fn

        img_acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        print(f"pred={n_pred}  gt={n_gt}  TP={n_tp}  FP={n_fp}  FN={n_fn}  acc={img_acc:.3f}")

        per_img_rows.append({
            "image": gj_path.stem,
            "n_pred": n_pred, "n_gt": n_gt,
            "TP": n_tp, "FP": n_fp, "FN": n_fn,
            "accuracy": round(img_acc, 4),
        })

    # ── Report globale ────────────────────────────────────────────────────────
    if not all_y_true:
        print("\n[ERROR] Nessun nucleo matchato. Controlla dist_thresh e i path.")
        sys.exit(1)

    print()
    print_and_save_report(all_y_true, all_y_pred, total_fp, total_fn,
                          args.output, args.dist_thresh)

    # ── Per-image CSV ─────────────────────────────────────────────────────────
    csv_path = args.output / "per_image.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image","n_pred","n_gt","TP","FP","FN","accuracy"]
        )
        writer.writeheader()
        writer.writerows(per_img_rows)
    print(f"  Per-image CSV → {csv_path}")


if __name__ == "__main__":
    main()
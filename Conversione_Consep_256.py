import argparse
import csv
import numpy as np
import scipy.io as sio
from collections import Counter
from pathlib import Path
import cv2
from skimage.color import rgb2hed
import os

# --------------------------------------------------------------------------- #
# Rimappatura CoNSeP (7 classi) → 4 classi come nel paper HoVer-Net
# --------------------------------------------------------------------------- #
CONSEP_REMAP = {
    0: 0,  # Background  → Background
    1: 3,  # Other       → Other
    2: 2,  # Inflammatory→ Inflammatory
    3: 3,  # Healthy Ep. → Other
    4: 1,  # Dyspl. Ep.  → Neoplastic
    5: 3,  # Fibroblast  → Other
    6: 3,  # Muscle      → Other
    7: 3,  # Endothelial → Other
}

PATCH_SIZE = 256

# Nomi classi dopo remap
CLASS_NAMES_REMAP = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Other"
}

# Nomi classi originali CoNSeP
CLASS_NAMES_ORIGINAL = {
    1: "Other",
    2: "Inflammatory",
    3: "Healthy_Epithelial",
    4: "Dysplastic_Epithelial",
    5: "Fibroblast",
    6: "Muscle",
    7: "Endothelial",
}

def rgb_to_hematoxylin_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """
    Estrae il canle ematossilina forzando il colore blu/violaceo
    su sfondo bianco anziché virare al rosa.
    """
    img_rgb_correct = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
    if img_rgb_correct.max() > 1:
        img = img_rgb_correct.astype(np.float64) / 255.0
    else:
        img = img_rgb_correct.astype(np.float64)

    stains = rgb2hed(img)
    h_channel = stains[:, :, 0]
    
    h_min, h_max = h_channel.min(), h_channel.max()
    if h_max - h_min > 0:
        h_norm = (h_channel - h_min) / (h_max - h_min)
    else:
        h_norm = np.zeros_like(h_channel)
        
    h_inv = 1.0 - h_norm

    res_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    res_img[:, :, 0] = (40 + h_inv * 215).astype(np.uint8)  # R
    res_img[:, :, 1] = (40 + h_inv * 215).astype(np.uint8)  # G
    res_img[:, :, 2] = (120 + h_inv * 135).astype(np.uint8) # B

    return cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

def pad_to_multiple(img: np.ndarray, mask_padded_shape, is_mask=False, pad_value=0):
    """
    Aggiunge del padding a destra e in basso per raggiungere la dimensione target (es. 1024x1024)
    """
    h, w = img.shape[:2]
    th, tw = mask_padded_shape[:2]
    pad_h = th - h
    pad_w = tw - w
    
    if pad_h == 0 and pad_w == 0:
        return img
        
    if is_mask:
        if len(img.shape) == 2:
            return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        else:
            return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0]*img.shape[2])
    else:
        # Per le immagini reali usiamo un padding bianco (255) per non alterare lo sfondo dell'istologia
        return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[pad_value]*3)

def convert_and_patch_one(
    mat_path: Path,
    output_dir: Path,
    png_dir: Path,
    png_out: Path,
    remap: bool,
    tissue_type: str,
    types_rows: list,
    cell_count_rows: list,
    count_ids: list,
    count_cols: list
):
    mat = sio.loadmat(str(mat_path))

    inst_map  = mat["inst_map"].astype(np.int32)   
    type_map  = mat["type_map"].astype(np.int32)   
    inst_type = mat["inst_type"].flatten().astype(np.int32)  

    # Applica il remap alla type_map pixel-wise originale se richiesto
    if remap:
        remapped_type_map = np.zeros_like(type_map)
        for src, dst in CONSEP_REMAP.items():
            remapped_type_map[type_map == src] = dst
        type_map = remapped_type_map

    # Carica l'immagine corrispondente
    img_name = mat_path.stem + ".png"
    single_png_path = png_dir / img_name
    
    if not single_png_path.exists():
        print(f"  [WARN] PNG non trovato: {single_png_path}, salto il file.")
        return

    img_bgr = cv2.imread(str(single_png_path))
    if img_bgr is None:
        print(f"  [WARN] Impossibile leggere: {single_png_path.name}, salto.")
        return

    # Applica subito la ricolorazione ad ematossilina blu
    img_h = rgb_to_hematoxylin_rgb(img_bgr)

    # Calcola la dimensione con padding (multiplo di 256 superiore a 1000 -> 1024)
    orig_h, orig_w = img_bgr.shape[:2]
    target_h = int(np.ceil(orig_h / PATCH_SIZE) * PATCH_SIZE)
    target_w = int(np.ceil(orig_w / PATCH_SIZE) * PATCH_SIZE)
    padded_shape = (target_h, target_w)

    # Esegui il padding di sicurezza
    img_padded = pad_to_multiple(img_h, padded_shape, is_mask=False, pad_value=255)
    inst_padded = pad_to_multiple(inst_map, padded_shape, is_mask=True)
    type_padded = pad_to_multiple(type_map, padded_shape, is_mask=True)

    # Generazione delle Patch (es. da 1024x1024 usciranno 4x4 = 16 patch)
    patch_idx = 0
    for r in range(0, target_h, PATCH_SIZE):
        for c in range(0, target_w, PATCH_SIZE):
            
            # Slicing della patch 256x256
            patch_img = img_padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            patch_inst = inst_padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            patch_type = type_padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE]

            # Identifica quali nuclei unici sono presenti in QUESTA patch
            unique_instances = np.unique(patch_inst)
            unique_instances = unique_instances[unique_instances != 0] # Escludi background

            # Conta le classi dei nuclei presenti nella patch usando inst_type originale
            patch_labels = []
            for inst_id in unique_instances:
                idx = inst_id - 1 # inst_type è 1-indexed
                if idx < len(inst_type):
                    lbl = int(inst_type[idx])
                    if remap:
                        lbl = CONSEP_REMAP.get(lbl, 0)
                    if lbl != 0:
                        patch_labels.append(lbl)
            
            patch_counts = Counter(patch_labels)

            # Nome univoco per la patch
            patch_name = f"{mat_path.stem}_p_{r}_{c}.png"
            npy_name = f"{mat_path.stem}_p_{r}_{c}.npy"

            # Salva l'immagine della patch
            cv2.imwrite(str(png_out / patch_name), patch_img)

            # Salva il dizionario .npy della patch
            np.save(
                str(output_dir / npy_name),
                {"inst_map": patch_inst, "type_map": patch_type},
            )

            # Salva i dati per i CSV dedicati alle patch
            types_rows.append({"img": patch_name, "type": tissue_type})

            row = {"img": patch_name}
            row.update(
                {
                    (CLASS_NAMES_REMAP[i] if remap else CLASS_NAMES_ORIGINAL[i]): patch_counts.get(i, 0)
                    for i in count_ids
                }
            )
            cell_count_rows.append(row)
            patch_idx += 1

    print(f"  {mat_path.name} generato → {patch_idx} patch da 256x256 | Tessuto: {tissue_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Patching di file .mat CoNSeP a 256x256 + Conversione Ematossilina Blu"
    )
    parser.add_argument("--input_dir",          required=True, help="Cartella .mat ('labels')")
    parser.add_argument("--output_dir",         required=True, help="Cartella out per .npy e .csv")
    parser.add_argument("--png_dir",            required=True, help="Cartella .png originali 1000x1000")
    parser.add_argument("--png_out_dir",        required=True, help="Cartella out per patch .png")
    parser.add_argument("--remap",              action="store_true", help="Rimappa classi CoNSeP → HoVer-Net")
    parser.add_argument("--tissue_type",        default="", help="Tissue type (es. 'Colon')")
    parser.add_argument("--tissue_from_parent", action="store_true", help="Usa nome cartella padre come tessuto")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    png_path   = Path(args.png_dir)
    png_out    = Path(args.png_out_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_out.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(input_dir.glob("*.mat"))
    if not mat_files:
        print(f"Nessun file .mat trovato in {input_dir}")
        return

    if args.remap:
        count_cols = [name for _, name in sorted(CLASS_NAMES_REMAP.items())]
        count_ids  = sorted(CLASS_NAMES_REMAP.keys())
    else:
        count_cols = [name for _, name in sorted(CLASS_NAMES_ORIGINAL.items())]
        count_ids  = sorted(CLASS_NAMES_ORIGINAL.keys())

    types_rows = []
    cell_count_rows = []

    print(f"Inizio Patching di {len(mat_files)} file a 256x256...\n")

    for mat_path in mat_files:
        tissue = args.tissue_type
        if args.tissue_from_parent:
            tissue = mat_path.parent.name

        convert_and_patch_one(
            mat_path, output_dir, png_path, png_out, args.remap, tissue,
            types_rows, cell_count_rows, count_ids, count_cols
        )

    # Scrittura dei file CSV aggiornati con i nomi e i conteggi delle singole PATCH
    types_path = output_dir / "types.csv"
    with open(types_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["img", "type"])
        writer.writeheader()
        writer.writerows(types_rows)

    cc_path = output_dir / "cell_count.csv"
    with open(cc_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["img"] + count_cols)
        writer.writeheader()
        writer.writerows(cell_count_rows)

    print(f"\nPatching Completato con successo!")
    print(f"  CSV Generati in: {output_dir}")


if __name__ == "__main__":
    main()
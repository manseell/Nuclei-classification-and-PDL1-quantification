"""
run_inference_batch.py
──────────────────────────────────────────────────────────────────────────────
Lancia inference_patch.py su ogni immagine .png in una cartella,
producendo per ciascuna un file cells.geojson.

Struttura output:
  out_dir/
    geojsons/          ← cartella piatta pronta per eval_consep.py
      test_1.geojson
      test_2.geojson
      ...
    test_1/            ← output completo di inference_patch.py
      cell_detection/
        cells.geojson
        cells.json
        cell_detection.json
    test_2/
      ...

Uso:
  python run_inference_batch.py \\
      --model   /path/to/model.pth \\
      --img_dir /path/to/images/ \\
      --out_dir /path/to/inference_out/ \\
      --gpu     0 \\
      --geojson
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Inferenza NuLite in batch su cartella di immagini"
    )
    parser.add_argument("--model",         required=True, type=Path,
                        help="Percorso al file .pth del modello NuLite")
    parser.add_argument("--img_dir",       required=True, type=Path,
                        help="Cartella con le immagini di input (.png)")
    parser.add_argument("--out_dir",       required=True, type=Path,
                        help="Cartella di output per i risultati")
    parser.add_argument("--gpu",           type=int, default=0)
    parser.add_argument("--magnification", type=int, default=20)
    parser.add_argument("--geojson",       action="store_true",
                        help="Salva anche i file .geojson (necessario per eval_consep.py)")
    parser.add_argument("--ext",           default=".png",
                        help="Estensione delle immagini (default: .png)")
    args = parser.parse_args()

    img_files = sorted(args.img_dir.glob(f"*{args.ext}"))
    if not img_files:
        print(f"[ERROR] Nessuna immagine {args.ext} trovata in {args.img_dir}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Cartella piatta per i geojson → pronta per eval_consep.py
    geojson_flat_dir = args.out_dir / "geojsons"
    geojson_flat_dir.mkdir(exist_ok=True)

    # inference_patch.py deve essere nella stessa cartella di questo script
    inference_script = Path(__file__).parent / "inference_patch.py"
    if not inference_script.exists():
        print(f"[ERROR] inference_patch.py non trovato in {inference_script.parent}")
        sys.exit(1)

    print(f"Immagini trovate : {len(img_files)}")
    print(f"Modello          : {args.model}")
    print(f"Output           : {args.out_dir}")
    print(f"GeoJSON piatti   : {geojson_flat_dir}\n")

    ok     = 0
    errors = []

    for i, img_path in enumerate(img_files, 1):
        stem = img_path.stem
        print(f"[{i:3d}/{len(img_files)}] {img_path.name} ...", end=" ", flush=True)

        # Copia l'immagine in out_dir/<stem>/ perché inference_patch.py
        # salva i risultati nella stessa cartella dell'immagine
        img_work_dir = args.out_dir / stem
        img_work_dir.mkdir(parents=True, exist_ok=True)
        img_work = img_work_dir / img_path.name
        if not img_work.exists():
            shutil.copy2(img_path, img_work)

        # Costruisce il comando
        cmd = [
            sys.executable, str(inference_script),
            "--model",         str(args.model),
            "--image",         str(img_work),
            "--gpu",           str(args.gpu),
            "--magnification", str(args.magnification),
        ]
        if args.geojson:
            cmd.append("--geojson")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("ERRORE")
            errors.append((img_path.name, result.stderr.strip()))
            print(f"    {result.stderr.strip()[-300:]}")
            continue

        # Copia cells.geojson nella cartella piatta rinominandolo con lo stem dell'immagine
        geojson_src = img_work_dir / "cell_detection" / "cells.geojson"
        if geojson_src.exists():
            geojson_dst = geojson_flat_dir / (stem + ".geojson")
            shutil.copy2(geojson_src, geojson_dst)
            print(f"OK")
            ok += 1
        else:
            print("OK (nessun geojson — aggiungi --geojson)")
            ok += 1

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    print(f"\n── Completato: {ok}/{len(img_files)} OK ──")

    if errors:
        print(f"\nImmagini con errore ({len(errors)}):")
        for name, err in errors:
            print(f"  {name}:\n    {err[-200:]}")

    if ok > 0 and args.geojson:
        print(f"\nGeojson pronti in: {geojson_flat_dir}")
        print("Ora puoi lanciare:")
        print(f"  python eval_consep.py --geojson_dir {geojson_flat_dir} --npy_dir /path/to/npy/ --output metrics/")


if __name__ == "__main__":
    main()
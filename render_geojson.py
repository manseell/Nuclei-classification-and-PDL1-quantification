# Versione .py dell'ultima cella di geojsontopng.ipynb

import argparse
import json
import os
from collections import defaultdict
import matplotlib.axes
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import Image, ImageFile

# Permette il caricamento di immagini troncate
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    # Configurazione dei parametri di comando
    parser = argparse.ArgumentParser(
        description = "Renderizza poligoni GeoJSON sopra un'immagine PNG"
    )
    parser.add_argument(
        "--image",
        "-i",
        required = True,
        help = "Percorso dell'immagine PNG di input",
    )
    parser.add_argument(
        "--geojson",
        "-g",
        required = True,
        help = "Percorso del file GeoJSON di input",
    )
    parser.add_argument(
        "--output",
        "-o",
        required = False,
        help = "Percorso del file PNG di ouput", 
    )

    args = parser.parse_args()

    # Caricamento del file 
    if not os.path.exists(args.image):
        print(f"Errore: L'immagine {args.image} non esiste ")
        return
    if not os.path.exists(args.geojson):
        print(f"Errore: Il file GeoJSON {args.geojson} non esiste")
        return

    region_rgb = np.array(Image.open(args.image))

    with open(args.geojson, "r") as f:
        geojson = json.load(f)

    h, w = region_rgb.shape[:2]

    # Raggruppa poligoni per classe
    class_data = defaultdict(lambda: {"polygons": [], "color": None})

    total = 0
    for feature in geojson:
        name = feature["properties"]["classification"]["name"]
        color = [
            c / 255 for c in feature["properties"]["classification"]["color"]
        ]
        multipolygon = feature["geometry"]["coordinates"]

        for polygon_group in multipolygon:
            for ring in polygon_group:
                exterior_ring = polygon_group[0]
                coords = np.array(exterior_ring)
                if coords.ndim == 2 and len(coords) >= 3:
                    class_data[name]["polygons"].append(coords)
                    class_data[name]["color"] = color
                    total += 1

    print(f"Totale nuclei: {total}")
    for name, d in class_data.items():
        print(f"  {name}: {len(d['polygons'])} nuclei")

    # 3. Generazione del plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(region_rgb)

    legend_handles = []
    for name, d in class_data.items():
        color = d["color"]
        patches = []
        for coords in d["polygons"]:
            poly = Polygon(coords, closed=True)
            patches.append(poly)

        pc = PatchCollection(
            patches,
            facecolor="none",
            edgecolor=[color],
            linewidth=2.0,
            alpha=0.9,
        )
        ax.add_collection(pc)
        legend_handles.append(
            mpatches.Patch(edgecolor=color, facecolor="none", label=name)
        )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)
    ax.axis("off")
    plt.tight_layout()

    # 4. Gestione del salvataggio di output
    if args.output:
        output_path = args.output
    else:
        # Se non specificato, salva nella stessa cartella dell'immagine con un suffisso
        dir_name = os.path.dirname(args.image)
        output_path = os.path.join(
            dir_name, "patch_inference_result_boundaries.png"
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Output salvato in: {output_path}")

    # Nota: plt.show() potrebbe bloccarsi se eseguito in un ambiente headless (es. Docker/SSH senza X11)
    # plt.show()


if __name__ == "__main__":
    main()
    
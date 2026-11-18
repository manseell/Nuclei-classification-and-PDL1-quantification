# inference_patch.py
import torch
import numpy as np
from PIL import Image, ImageFile
from pathlib import Path
import argparse
import ujson
import uuid
import pandas as pd
import torch.nn.functional as F
import io as _io

from nuclei_detection.inference.nuclei_detection import CellSegmentationInference, TYPE_NUCLEI_DICT, COLOR_DICT
from nuclei_detection.utils.template_geojson import get_template_segmentation, get_template_point

ImageFile.LOAD_TRUNCATED_IMAGES = True

def run_patch_inference(model_path, image_path, gpu=0, geojson=False, magnification=20):
    cell_seg = CellSegmentationInference(model_path=model_path, gpu=gpu)

    # Carica la patch
    with open(image_path, "rb") as _f:
        _raw = _io.BytesIO(_f.read())
    _tmp = Image.open(_raw)
    _tmp.info.pop("icc_profile", None)
    img = _tmp.convert("RGB")
    patch_np = np.array(img)

    # Applica le trasformazioni (ToTensor + Normalize)
    patch_tensor = cell_seg.inference_transforms(patch_np)  # shape: (3, H, W)
    patch_tensor = patch_tensor.unsqueeze(0).to(cell_seg.device)  # shape: (1, 3, H, W)
        
    
    dataset_config = cell_seg.run_conf["dataset_config"]
    nuclei_types = dataset_config["nuclei_types"]

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predictions = cell_seg.model.forward(patch_tensor, retrieve_tokens=True)

    instance_types, tokens = cell_seg.get_cell_predictions_with_tokens(
        predictions, magnification=magnification
    )

    # Estrai celle dalla prima (unica) patch del batch
    patch_instance_types = instance_types[0]

    cell_dict_wsi = []
    cell_dict_detection = []

    for cell in patch_instance_types.values():
        if cell["type"] == nuclei_types["Background"]:
            continue
        cell_dict = {
            "bbox": cell["bbox"].tolist(),
            "centroid": cell["centroid"].tolist(),
            "contour": cell["contour"].tolist(),
            "type_prob": cell["type_prob"],
            "type": cell["type"],
        }
        cell_detection = {
            "bbox": cell["bbox"].tolist(),
            "centroid": cell["centroid"].tolist(),
            "type": cell["type"],
        }
        cell_dict_wsi.append(cell_dict)
        cell_dict_detection.append(cell_detection)

    print(f"Detected cells: {len(cell_dict_wsi)}")

    # Salva output
    outdir = Path(image_path).parent / "cell_detection"
    outdir.mkdir(exist_ok=True, parents=True)

    with open(str(outdir / "cells.json"), "w") as f:
        ujson.dump({"type_map": nuclei_types, "cells": cell_dict_wsi}, f, indent=2)

    with open(str(outdir / "cell_detection.json"), "w") as f:
        ujson.dump({"type_map": nuclei_types, "cells": cell_dict_detection}, f, indent=2)

    if geojson:
        geojson_list = cell_seg.convert_geojson(cell_dict_wsi, polygons=True)
        with open(str(outdir / "cells.geojson"), "w") as f:
            ujson.dump(geojson_list, f, indent=2)
        geojson_det = cell_seg.convert_geojson(cell_dict_wsi, polygons=False)
        with open(str(outdir / "cell_detection.geojson"), "w") as f:
            ujson.dump(geojson_det, f, indent=2)
        print(f"GeoJSON saved in: {outdir}")

    print(f"Results saved in: {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--geojson", action="store_true")
    parser.add_argument("--magnification", type=int, default=20)
    args = parser.parse_args()

    run_patch_inference(args.model, args.image, args.gpu, args.geojson, args.magnification)
# -*- coding: utf-8 -*-
# Prepare MoNuSeg Dataset By converting and resorting files
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
from nuclei_detection.utils.metrics import remap_label


def process_fold(fold, input_path, output_path) -> None:
    fold_path = Path(input_path) / f"fold{fold}"
    output_fold_path = Path(output_path) / f"fold{fold}"
    output_fold_path.mkdir(exist_ok=True, parents=True)
    (output_fold_path / "images").mkdir(exist_ok=True, parents=True)
    (output_fold_path / "labels").mkdir(exist_ok=True, parents=True)

    print(f"Fold: {fold}")
    print("Loading large numpy files, this may take a while")
    images = np.load(fold_path / "images.npy")
    masks = np.load(fold_path / "masks.npy")

    print("Process images")
    for i in tqdm(range(len(images)), total=len(images)):
        outname = f"{fold}_{i}.png"
        img_rgb = images[i]
        # Conversione RGB -> HEM
        ihc_hed = rgb2hed(img_rgb)
        h_channel = ihc_hed[:, :, 0]
        h_min, h_max = h_channel.min(), h_channel.max()
        h_normalized = (255 * (h_channel - h_min) / (h_max - h_min + 1e-7)).astype(np.uint8)
        
        im = Image.fromarray(h_normalized)
        im.save(output_fold_path / "images" / outname)

    print("Process masks")
    for i in tqdm(range(len(images)), total=len(images)):
            outname = f"{fold}_{i}.npy"
            mask = masks[i] # Shape (256, 256, 5)
            
            # Gestione Instance Map (rimane simile, ma dobbiamo unificare le istanze)
            inst_map = np.zeros((256, 256), dtype=np.int32)
            num_nuc = 0
            for j in range(5):
                layer_res = remap_label(mask[:, :, j])
                inst_map = np.where(layer_res != 0, layer_res + num_nuc, inst_map)
                num_nuc += np.max(layer_res)
            inst_map = remap_label(inst_map)
    
            # 2. Gestione Type Map (La tua logica 4-classi)
            type_map = np.zeros((256, 256), dtype=np.int32)
            for j in range(5):
                # j=0: Neoplastic -> Classe 1
                # j=1: Inflammatory -> Classe 2
                # j=2,3,4: Altri -> Classe 3
                if j == 0:
                    new_class = 1
                elif j == 1:
                    new_class = 2
                else:
                    new_class = 3
                    
                layer_mask = (mask[:, :, j] > 0).astype(np.int32)
                # Sovrascriviamo il type_map solo dove c'è un nucleo
                type_map = np.where(layer_mask != 0, new_class, type_map)
    
            outdict = {"inst_map": inst_map, "type_map": type_map}
            np.save(output_fold_path / "labels" / outname, outdict)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Perform CellViT inference for given run-directory with model checkpoints and logs",
)
parser.add_argument(
    "--input_path",
    type=str,
    help="Input path of the original PanNuke dataset",
    required=True,
)
parser.add_argument(
    "--output_path",
    type=str,
    help="Output path to store the processed PanNuke dataset",
    required=True,
)

if __name__ == "__main__":
    opt = parser.parse_args()
    configuration = vars(opt)

    input_path = Path(configuration["input_path"])
    output_path = Path(configuration["output_path"])
    #input_path = '/work/oldWork/default-ubuntu-rinaldi2-work-pvc-215307e1-04be-4abb-8a29-26eb803ad2b3/cristian/data/Pannuke'
    #output_path = '/work/cristian/data/prepared_pannuke'
    for fold in [0, 1, 2]:
        process_fold(fold, input_path, output_path)

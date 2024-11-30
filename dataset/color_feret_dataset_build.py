import argparse
import os
import random
import itertools
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from pathlib import Path

def get_label_from_filename(filename):
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')
    label = '_'.join(parts[:2])
    return label

if __name__=="__main__":
    parser = argparse.ArgumentParser("Create FEI FatialDataSet for Testing")
    parser.add_argument("--color_dir", type=str, default="/mnt/e/Downloads/colorferet/images_retina",)
    parser.add_argument("--output_file", type=str, default="/mnt/e/Downloads/colorferet/images_retina/labels.csv",)

    args = parser.parse_args()

    issame_file_path = Path(args.output_file)
    issame_list = []
    iter = 0

    color_dir = Path(args.color_dir)
    subdirs = [x for x in color_dir.iterdir() if x.is_dir()]

    for subdir in subdirs:
        subsubdirs = [x for x in subdir.iterdir() if x.is_dir()]
        for subsubdir in tqdm(subsubdirs):
            jpg_files = list(subsubdir.glob("*.jpg"))
            labels = set(get_label_from_filename(jpg_file.name) for jpg_file in jpg_files)
            for label in labels:
                fa_files = list(subsubdir.glob(f"{label}_fa*.jpg"))
                fb_files = list(subsubdir.glob(f"{label}_fb*.jpg"))
                hl_files = list(subsubdir.glob(f"{label}_hl*.jpg"))
                hr_files = list(subsubdir.glob(f"{label}_hr*.jpg"))
                
                if fa_files and fb_files and hl_files and hr_files:
                    for file in fa_files + fb_files + hl_files + hr_files:
                        issame_list.append((iter, str(file.relative_to(issame_file_path.parent)), label))
                        iter += 1
            
    with open(args.output_file, 'w') as f:
        for item in tqdm(issame_list, desc="Writing to labels.csv"):
            f.write("{},{},{}\n".format(item[0], item[1], item[2]))
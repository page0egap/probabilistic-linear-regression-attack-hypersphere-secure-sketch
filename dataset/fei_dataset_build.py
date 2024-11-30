import argparse
import os
import random
import itertools
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from pathlib import Path

if __name__=="__main__":
    parser = argparse.ArgumentParser("Create FEI FatialDataSet for Testing")
    parser.add_argument("--fei_dir", type=str, default="/mnt/e/Downloads/FEI/originalimages_retinaface_122/originalimages",)
    parser.add_argument("--output_file", type=str, default="/mnt/e/Downloads/FEI/originalimages_retinaface_122/labels.csv",)

    args = parser.parse_args()

    issame_file_path = Path(args.output_file)
    issame_list = []

    # only use 04, 05, 06
    iter = 0
    for i in tqdm(range(1, 201), desc="Translating"):
        for j in [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]:  # differect poses we want to include in our dataset
            image_path_0 = Path(args.fei_dir, "{}-{}.jpg".format(i, str(j).zfill(2)))
            image_relative_path_0 = image_path_0.relative_to(issame_file_path.parent)
            issame_list.append([iter, str(image_relative_path_0), i])
            iter += 1

    with open(args.output_file, 'w') as f:
        for item in tqdm(issame_list, desc="Writing to labels.csv"):
            f.write("{},{},{}\n".format(item[0], item[1], item[2]))
import argparse
import os
import random
import itertools
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from pathlib import Path

def get_label_from_filename(filename):
    # 获取文件名（不包括扩展名）
    base_name = os.path.splitext(filename)[0]
    # 以下划线分割文件名
    parts = base_name.split('_')
    # 获取第二个下划线之前的部分作为label
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
        # 进入当前文件夹下的所有子文件夹
        subsubdirs = [x for x in subdir.iterdir() if x.is_dir()]
        for subsubdir in tqdm(subsubdirs):
            # 获得当前目录下的所有后缀为jpg的文件名称
            jpg_files = list(subsubdir.glob("*.jpg"))
            # 获得jpg_file第二个下划线之前的部分作为label并去重
            labels = set(get_label_from_filename(jpg_file.name) for jpg_file in jpg_files)
            # 如果当前label有形如形式为label_fa*, label_fb*, label_hl*, label_hr*所有文件存在，则将它们加入issame_list, 形式为三元组(iter, path_relative_to_issame_file_path's parent, label)
            for label in labels:
                fa_files = list(subsubdir.glob(f"{label}_fa*.jpg"))
                fb_files = list(subsubdir.glob(f"{label}_fb*.jpg"))
                hl_files = list(subsubdir.glob(f"{label}_hl*.jpg"))
                hr_files = list(subsubdir.glob(f"{label}_hr*.jpg"))
                
                if fa_files and fb_files and hl_files and hr_files:
                    # 将所有files加入issame_list中
                    for file in fa_files + fb_files + hl_files + hr_files:
                        issame_list.append((iter, str(file.relative_to(issame_file_path.parent)), label))
                        iter += 1
            
    with open(args.output_file, 'w') as f:
        for item in tqdm(issame_list, desc="Writing to labels.csv"):
            f.write("{},{},{}\n".format(item[0], item[1], item[2]))
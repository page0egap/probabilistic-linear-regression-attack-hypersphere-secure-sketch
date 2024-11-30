# add retinaface path to sys.path
import sys
sys.path.append("../insightface/detection/retinaface")
from retinaface import RetinaFace

import argparse
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np

import face_align

target_size = 400
max_size = 800

def get_norm_crop(image_path):
    global detector
    global args
    im = cv2.imread(image_path)
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    bbox, landmark = detector.detect(im, threshold=0.5, scales=[im_scale])
    #print(im.shape, bbox.shape, landmark.shape)
    if bbox.shape[0] == 0:
        bbox, landmark = detector.detect(
            im,
            threshold=0.05,
            scales=[im_scale * 0.75, im_scale, im_scale * 2.0])
        print('refine', im.shape, bbox.shape, landmark.shape)
    nrof_faces = bbox.shape[0]
    if nrof_faces > 0:
        det = bbox[:, 0:4]
        img_size = np.asarray(im.shape)[0:2]
        bindex = 0
        if nrof_faces > 1:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                           det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            bindex = np.argmax(bounding_box_size - offset_dist_squared *
                               2.0)  # some extra weight on the centering
        #_bbox = bounding_boxes[bindex, 0:4]
        _landmark = landmark[bindex]
        warped = face_align.norm_crop(im,
                                      landmark=_landmark,
                                      image_size=args.image_size,
                                      mode=args.align_mode)
        return warped
    else:
        return None


if __name__=="__main__":
    parser = argparse.ArgumentParser("Align Face Dataset Using RetinaFace", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--data_dir", type=str, default="/mnt/e/Downloads/colorferet/dvd2/data/images",)
    parser.add_argument("--output_dir", type=str, default="/mnt/e/Downloads/colorferet/dvd2conda/data/images_retina",)
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--gpu", type=int, default=-1, help="Do not use gpu, set default -1")
    parser.add_argument("--det_prefix", type=str, default="/mnt/e/BaiduNet/retinaface-R50/R50")
    parser.add_argument("--align_mode", type=str, default="arcface")

    
    args = parser.parse_args()

    gpu_id = args.gpu

    detector = RetinaFace(args.det_prefix, 0, gpu_id, 'net3')

    # open data_dir path and get all subdirs and files then use Retinaface to crop face and save to output_dir
    # use threading to speed up cv2.imwrite

    for subdir in (pbar:=tqdm(os.listdir(args.data_dir), desc="Progressing")):
        with ProcessPoolExecutor(max_workers=8) as executor:
            subdir_path = os.path.join(args.data_dir, subdir)
            if os.path.isdir(subdir_path):
                for img in os.listdir(subdir_path):
                    img_path = os.path.join(subdir_path, img)
                    if not img_path.endswith(".jpg"):
                        continue
                    img_cropped = get_norm_crop(img_path)
                    if img_cropped is not None:
                        output_path = os.path.join(args.output_dir, subdir)
                        if not os.path.exists(output_path):
                            os.makedirs(output_path)
                        # store cv2 format image to jpg
                        executor.submit(cv2.imwrite, os.path.join(output_path, img), img_cropped)
                        # cv2.imwrite(os.path.join(output_path, img), img_cropped)
            else:
                img = subdir
                img_path = subdir_path
                if not img_path.endswith(".jpg"):
                    continue
                img_cropped = get_norm_crop(img_path)
                if img_cropped is not None:
                    output_path = args.output_dir
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    # store cv2 format image to jpg
                    executor.submit(cv2.imwrite, os.path.join(output_path, img), img_cropped)
                    # cv2.imwrite(os.path.join(output_path, img), img_cropped)


    
    




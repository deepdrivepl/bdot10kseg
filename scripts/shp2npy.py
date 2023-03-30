import os
import sys
import time
import random
import math
sys.path.append("../bdot10kseg")

from semseg_dataset import BDOT10kSemSeg

from glob import glob
from tqdm import tqdm

import numpy as np
import cv2


SIZE = 1024
IMG_LIST = "../data/bdot10kseg-463.txt"
CAT_CSV = "../data/BDOT10k-categories.csv"
TIF_DIR = "../dataset/images"
SHP_DIR = "../dataset/SHP"
POW_SHP = "../dataset/powiaty/powiaty.shp"
OUT_DIR = "../dataset/data/463-%d" % SIZE

if __name__ == "__main__":
    flist = [os.path.join(TIF_DIR, x.rstrip()) for x in open(IMG_LIST)]

    img_dir = os.path.join(OUT_DIR, "images")
    lbl_dir = os.path.join(OUT_DIR, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)


    dataset = BDOT10kSemSeg(
         tiff_dir=flist,
         shp_dir=SHP_DIR,
         powiaty_shp_fname=POW_SHP,
         bdot10k_cats_fname=CAT_CSV,
         level=0,
         classes=None
    )


    for b in tqdm(dataset):
        image, mask, pth = b
        H,W = mask.shape[1], mask.shape[2]
        N,M = math.floor(H/SIZE), math.floor(W/SIZE)

        for n in range(N):
            for m in range(M):
                y0, y1 = n*SIZE, (n+1)*SIZE
                x0, x1 = m*SIZE, (m+1)*SIZE
                m = mask[:,y0:y1, x0:x1]

                lbl_path = os.path.join(lbl_dir, os.path.splitext(pth)[0]+"_%d_%d.npy" % (x0, y0))
                img_path = os.path.join(img_dir, os.path.splitext(pth)[0]+"_%d_%d.jpg" % (x0, y0))

                if np.all(m[1,:,:]==0) and np.all(m[2,:,:]==0):
                    if random.randint(0,100) < 5:
                        im = image[y0:y1,x0:x1,:]
                        cv2.imwrite(img_path, im[:,:,::-1])
                        with open(lbl_path, 'wb') as f:
                            np.save(f, m)
                else:
                    im = image[y0:y1,x0:x1,:]
                    cv2.imwrite(img_path, im[:,:,::-1])
                    with open(lbl_path, 'wb') as f:
                        np.save(f, m)  
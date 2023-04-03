from rasterio.features import rasterize
import random
import math
import geopandas as gpd
import cv2
import numpy as np
import matplotlib
import torch
import rasterio
import pandas as pd
from collections import defaultdict
from glob import glob
import shapely
import io
import functools
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

from base_dataset import BDOT10kDataset


class BDOT10kDetection(BDOT10kDataset):
    """BDOT10k dataset"""
    def __init__(self, **kwargs):  
        super(BDOT10kDetection, self).__init__(**kwargs)
        self.colors = self.get_colors()


    def get_label(self, df, idx):
        
        def check_bbox(x0,y0,x1,y1,W,H):
            valid = True
            x0,y0,x1,y1 = [0 if x<0 else x for x in [x0,y0,x1,y1]]
            x0,x1 = [W-1 if x>W-1 else x for x in [x0,x1]]
            y0,y1 = [H-1 if x>H-1 else x for x in [y0,y1]]
            
            w, h = x1-x0, y1-y0
            if w == 0 or h == 0:
                valid = False
            return valid,x0,y0,x1,y1
                
        
        img_box = shapely.geometry.box(0, 0, self.W-1, self.H-1)
        cat_names = list(self.bdot10k_cats_dict.keys())

        classes, bboxes, areas = [], [], []
        if df is not None:
            df_ = df[["geometry", "X_KOD"]].dropna()

            for i in range(len(df_)):
                if self.classes is not None and not df_.iloc[i].X_KOD.startswith(tuple(self.classes)):
                    continue

                try:
                    geom = df_.iloc[i].geometry.intersection(img_box)
                    if geom.is_empty:
                        continue
                except:
                    continue

                
                _x0,_y0,_x1,_y1 = list(map(int, geom.bounds))
                _x0,_x1 = min(_x0,_x1), max(_x0,_x1)
                _y0,_y1 = min(_y0,_y1), max(_y0,_y1)
                valid,_x0,_y0,_x1,_y1 = check_bbox(_x0,_y0,_x1,_y1,self.W,self.H)
                if not valid:
                    continue

                xc,yc,w,h = (_x0+_x1)//2, (_y0+_y1)//2, _x1-_x0, _y1-_y0

                classes.append(cat_names.index(df_.iloc[i].X_KOD[:self.level["trunc"]]))
                bboxes.append([xc,self.H-yc,w,h]) # quick fix
                areas.append((_x1-_x0)*(_y1-_y0))

        label = {
            "boxes": np.array(bboxes),
            "labels": np.array(classes),
            "areas": np.array(areas),
            "iscrowd": np.zeros((np.array(bboxes).shape[0],)),
            "image_id": np.array([idx])
        }

        return label

    
    def get_colors(self):
        palette = itertools.cycle(sns.color_palette())
        
        colors = {}
        for i, (_, clr) in enumerate(zip(self.bdot10k_cats_dict.keys(), palette)):
            colors[i] = tuple([int(x*255) for x in clr])
        return colors
        
    
    def plot_sample(self, img, mask, show=False):

        if mask["boxes"].shape[0] != 0:

            for cls, bbox in zip(mask["labels"], mask["boxes"]):
                xc,yc,w,h = bbox
                xmin, ymin = int(xc-0.5*w), int(yc-0.5*h)
                xmax, ymax = int(xc+0.5*w), int(yc+0.5*h)

                img = cv2.rectangle(img.copy(), (xmin, ymin), (xmax, ymax), self.colors[int(cls.item())], thickness=3)
        
        fig = plt.figure(figsize=(15,15))
        plt.imshow(img[:,:,::-1])
        plt.title(" ".join(self.bdot10k_cats_dict.keys()))
        
        if show:
            plt.show()
        else:
            with io.BytesIO() as buff:
                fig.savefig(buff, format='raw')
                buff.seek(0)
                data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            img = data.reshape((int(h), int(w), -1))
            return img

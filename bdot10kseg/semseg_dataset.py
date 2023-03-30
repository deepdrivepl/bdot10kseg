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
import matplotlib.pyplot as plt

from base_dataset import BDOT10kDataset, BDOT10kDatasetOrig


class BDOT10kSemSegOrig(BDOT10kDatasetOrig):
    """BDOT10k dataset"""
    def __init__(self, binary=False, **kwargs):  
        self.binary = binary
        super(BDOT10kSemSegOrig, self).__init__(**kwargs)

    def get_label(self, df, idx):
        mask = np.zeros((self.get_heads_count(), self.H, self.W), dtype='int')#self.size, self.size), dtype='int')
        cat_names = list(self.bdot10k_cats_dict.keys())
        if df is not None:
            for code in df.X_KOD.unique():
                if self.classes is not None and not code.startswith(tuple(self.classes)):
                    continue
                out_id = cat_names.index(code[:self.level["trunc"]])
                m = rasterize(shapes=df[df.X_KOD==code].geometry.iloc, out_shape=(self.H, self.W))[::-1]
                mask[out_id][m!=0] = m[m!=0]
        return mask
    
    def plot_sample(self, img, mask, show=False):
        cat_names = list(self.bdot10k_cats_dict.keys())
        
        n, m = 5, int(math.ceil((len(self.bdot10k_cats_dict)+1)/5))
        fig = plt.figure(figsize=(n*5, m*5))
        plt.subplot(m,n,1)
        plt.imshow(img[:,:,::-1])
        
        for i,vmax in enumerate([len(_) for _ in self.bdot10k_cats_dict.values()]):
            plt.subplot(m,n,2+i)
            # print(np.unique(mask[i]))
            plt.imshow(mask[i], cmap='gray')#,cmap='gist_ncar',vmin=0, vmax=vmax)
            plt.axis('off')
            plt.title(cat_names[i]+'\n'+ self.bdot10k_df[self.bdot10k_df[self.level["code"]]==cat_names[i]].iloc[0][self.level["name"]])
        plt.tight_layout()
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


class BDOT10kSemSeg(BDOT10kDataset):
    """BDOT10k dataset"""
    def __init__(self, binary=False, **kwargs):  
        self.binary = binary
        super(BDOT10kSemSeg, self).__init__(**kwargs)

    def get_label(self, df, idx):
        mask = np.zeros((self.get_heads_count(), self.H, self.W), dtype='int')#self.size, self.size), dtype='int')
        cat_names = list(self.bdot10k_cats_dict.keys())
        if df is not None:
            for code in df.X_KOD.unique():
                if self.classes is not None and not code.startswith(tuple(self.classes)):
                    continue
                out_id = cat_names.index(code[:self.level["trunc"]])
                m = rasterize(shapes=df[df.X_KOD==code].geometry.iloc, out_shape=(self.H, self.W))[::-1]
                mask[out_id][m!=0] = m[m!=0]
        return mask
    
    def plot_sample(self, img, mask, show=False):
        cat_names = list(self.bdot10k_cats_dict.keys())
        
        n, m = 5, int(math.ceil((len(self.bdot10k_cats_dict)+1)/5))
        fig = plt.figure(figsize=(n*5, m*5))
        plt.subplot(m,n,1)
        plt.imshow(img[:,:,::-1])
        
        for i,vmax in enumerate([len(_) for _ in self.bdot10k_cats_dict.values()]):
            plt.subplot(m,n,2+i)
            # print(np.unique(mask[i]))
            plt.imshow(mask[i], cmap='gray')#,cmap='gist_ncar',vmin=0, vmax=vmax)
            plt.axis('off')
            plt.title(cat_names[i]+'\n'+ self.bdot10k_df[self.bdot10k_df[self.level["code"]]==cat_names[i]].iloc[0][self.level["name"]])
        plt.tight_layout()
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

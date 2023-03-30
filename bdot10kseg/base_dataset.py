from rasterio.features import rasterize
import random
import math
import os
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


class BDOT10kDataset(torch.utils.data.Dataset):
    """BDOT10k dataset"""
    def __init__(self,
                 tiff_dir,
                 shp_dir,
                 powiaty_shp_fname,
                 bdot10k_cats_fname,
                 size=1024,
                 level=0,
                 transform=None,
                 classes=None):
        self.tiff_dir = tiff_dir
        self.tiffs = tiff_dir #sorted(glob(tiff_dir+'/*.tif'))
        print(f'Found {len(self.tiffs)} tiff files') # in {self.tiff_dir}')
        self.powiaty = gpd.read_file(powiaty_shp_fname)
        print(f'Found {len(self.powiaty)} powiats')
        
        self.shps = sorted(glob(shp_dir+'/*.shp'))
        print(f'SHPs: {len(self.shps)}')
        
        self.bdot10k_df = pd.read_csv(bdot10k_cats_fname)
        self.bdot10k_cats = sorted(self.bdot10k_df.kod2.tolist())
        self.bdot10k_cats_dict = defaultdict(list)
        
        self.level = self.get_level_info(level)
        self.classes = classes
        for k in self.bdot10k_cats:
            if self.classes is None:
                self.bdot10k_cats_dict[k[:self.level["trunc"]]].append(k)
            else:
                if k.startswith(tuple(classes)):
                    self.bdot10k_cats_dict[k[:self.level["trunc"]]].append(k)
                
        print(f'Classes: {len(self.bdot10k_cats_dict)}')
        for k in self.bdot10k_cats_dict:
            print(f'{k}:', self.bdot10k_cats_dict[k])
        
        self.transform = transform
        self.size = size
        

    def __len__(self):
        return len(self.tiffs)
    
    
    def get_level_info(self, level):
        levels = {
            0: {
                "trunc": 2,
                "code": "kod0",
                "name": "kategoria"
            },
            1: {
                "trunc": 4,
                "code": "kod1",
                "name": "klasa"
            },
            2: {
                "trunc": None,
                "code": "kod2",
                "name": "obiekt"
            },
        }
        return levels[level]
        

    def get_tile(self, raster):
        img = raster.read().transpose(1,2,0)

        # x0 = random.randint(0,img.shape[1]-self.size-1)
        # y0 = random.randint(0,img.shape[0]-self.size-1)
        # x1 = x0+self.size
        # y1 = y0+self.size
        
        x0 = 0
        y0 = 0
        x1 = x0+self.size
        y1 = y0+self.size
        
        _x0, _y0 = raster.transform * (x0,y0)
        _x1, _y1 = raster.transform * (x1,y1)
        
        _x0,_x1 = min(_x0,_x1), max(_x0,_x1)
        _y0,_y1 = min(_y0,_y1), max(_y0,_y1)
        
        return img[y0:y1,x0:x1], (_x0,_y0), (_x1,_y1) # crop coords in meters
    
    def get_id_by_code(self, code):
        return self.bdot10k_cats.index(code)
    
#     @functools.lru_cache(maxsize=10)
    def load_tiff(self, idx):
        tiff_fname = self.tiffs[idx]
        raster = rasterio.open(tiff_fname)
        return raster
    
    def get_image(self, idx):
        raster = self.load_tiff(idx)
        img, (x0,y0),(x1,y1) = self.get_tile(raster)
        return img, x0,y0,x1,y1
        
    def get_heads_count(self):
        return len(self.bdot10k_cats_dict.keys())

    def get_label(self, df, idx):
        raise NotImplementedError
        
    @functools.lru_cache(maxsize=100)
    def load_bdot10k_for_powiat(self, powiat_id):
        df_to_merge: list = []
        for fn in self.shps:
            if powiat_id in fn:
                df = gpd.read_file(fn)
                df_to_merge.append(df)
        if len(df_to_merge) > 0:
            df_merged = pd.concat(df_to_merge)
            return df_merged
        else:
            return None
    
    def load_bdot10k_for_powiats(self, powiat_ids):
        df_to_merge: list = []
        for powiat_id in powiat_ids:
            df = self.load_bdot10k_for_powiat(powiat_id)
            df_to_merge.append(df)
        if len(df_to_merge)>0:
            df_merged = pd.concat(df_to_merge)
            return df_merged
        else:
            return None
    
    def powiat_ids_for_bbox(self, x0, y0, x1, y1):
        bbox = shapely.geometry.box(x0, y0, x1, y1)

        powiat_ids = []
        for p in self.powiaty.iloc:
            if p.geometry.contains(bbox) or bbox.contains(p.geometry):
                powiat_ids.append(p.JPT_KOD_JE)
        return powiat_ids
    
    def plot_sample(self, img, mask, show=False):
        raise NotImplementedError

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img, x0,y0,x1,y1 = self.get_image(idx)
        powiat_ids = self.powiat_ids_for_bbox(x0,y0,x1,y1)
        
        
        df = self.load_bdot10k_for_powiats(powiat_ids)
        if df is not None:
            df = df.cx[x0:x1, y0:y1]
            new_geoms = []
            for i in range(len(df)):
                s = df.iloc[i].geometry
                if 'MultiPolygon' in str(type(df.iloc[i].geometry)):
                    ps = []
                    for p in df.iloc[i].geometry:
                        xys = np.array([_ for _ in zip(*p.exterior.xy)])-(x0, y0)
                        xys[:,0]=xys[:,0]*img.shape[1]/(x1-x0)
                        xys[:,1]=xys[:,1]*img.shape[0]/(y1-y0)
                        ps.append(shapely.geometry.Polygon(xys))
                    s = shapely.geometry.MultiPolygon(ps)
                elif 'Polygon' in str(type(df.iloc[i].geometry)):
                    xys = np.array([_ for _ in zip(*df.iloc[i].geometry.exterior.xy)])-(x0, y0)
                    xys[:,0]=xys[:,0]*img.shape[1]/(x1-x0)
                    xys[:,1]=xys[:,1]*img.shape[0]/(y1-y0)
                    s = shapely.geometry.Polygon(xys)
                elif 'LineString' in str(type(df.iloc[i].geometry)):
                    xys = np.array(df.iloc[i].geometry.coords)-(x0, y0)
                    xys[:,0]=xys[:,0]*img.shape[1]/(x1-x0)
                    xys[:,1]=xys[:,1]*img.shape[0]/(y1-y0)
                    s = shapely.geometry.LineString(xys)
                elif 'MultiPoint' in str(type(df.iloc[i].geometry)):
                    ps = []
                    for p in df.iloc[i].geometry.geoms:
                        x = (p.x-x0)*img.shape[1]/(x1-x0)
                        y = (p.y-y0)*img.shape[0]/(y1-y0)
                        ps.append((x,y))
                    s = shapely.geometry.MultiPoint(ps)
                elif 'Point' in str(type(df.iloc[i].geometry)):
                    p = df.iloc[i].geometry
                    x = (p.x-x0)*img.shape[1]/(x1-x0)
                    y = (p.y-y0)*img.shape[0]/(y1-y0)
                    s = shapely.geometry.Point(x,y)
                else:
                    print(type(df.iloc[i].geometry))
                new_geoms.append(s)

            df.geometry = new_geoms
        else:
            df = None
                   
        label = self.get_label(df, idx)
        return img, label



class BDOT10kDatasetOrig(torch.utils.data.Dataset):
    """BDOT10k dataset"""
    def __init__(self,
                 tiff_dir,
                 shp_dir,
                 powiaty_shp_fname,
                 bdot10k_cats_fname,
                 level=0,
                 transform=None,
                 classes=None):
        # self.tiff_dir = tiff_dir
        self.tiffs = tiff_dir #sorted(glob(tiff_dir+'/*.tif'))
        print(f'Found {len(self.tiffs)}')
        self.powiaty = gpd.read_file(powiaty_shp_fname)
        print(f'Found {len(self.powiaty)} powiats')
        
        self.shps = sorted(glob(shp_dir+'/*.shp'))
        print(f'SHPs: {len(self.shps)}')
        
        self.bdot10k_df = pd.read_csv(bdot10k_cats_fname)
        self.bdot10k_cats = sorted(self.bdot10k_df.kod2.tolist())
        self.bdot10k_cats_dict = defaultdict(list)
        
        self.level = self.get_level_info(level)
        self.classes = classes
        for k in self.bdot10k_cats:
            if self.classes is None:
                self.bdot10k_cats_dict[k[:self.level["trunc"]]].append(k)
            else:
                if k.startswith(tuple(classes)):
                    self.bdot10k_cats_dict[k[:self.level["trunc"]]].append(k)
                
        print(f'Classes: {len(self.bdot10k_cats_dict)}')
        
        self.transform = transform
        

    def __len__(self):
        return len(self.tiffs)
    
    
    def get_level_info(self, level):
        levels = {
            0: {
                "trunc": 2,
                "code": "kod0",
                "name": "kategoria"
            },
            1: {
                "trunc": 4,
                "code": "kod1",
                "name": "klasa"
            },
            2: {
                "trunc": None,
                "code": "kod2",
                "name": "obiekt"
            },
        }
        return levels[level]
        

    def get_tile(self, raster):
        img = raster.read().transpose(1,2,0)
        self.H, self.W = raster.height, raster.width
        
        x0 = 0
        y0 = 0
        x1 = x0+self.W-1
        y1 = y0+self.H-1
        
        _x0, _y0 = raster.transform * (x0,y0)
        _x1, _y1 = raster.transform * (x1,y1)
        
        _x0,_x1 = min(_x0,_x1), max(_x0,_x1)
        _y0,_y1 = min(_y0,_y1), max(_y0,_y1)
        
        return img[y0:y1,x0:x1], (_x0,_y0), (_x1,_y1) # crop coords in meters
    
    def get_id_by_code(self, code):
        return self.bdot10k_cats.index(code)
    
#     @functools.lru_cache(maxsize=10)
    def load_tiff(self, idx):
        tiff_fname = self.tiffs[idx]
        raster = rasterio.open(tiff_fname)
        return raster
    
    def get_image(self, idx):
        raster = self.load_tiff(idx)
        img, (x0,y0),(x1,y1) = self.get_tile(raster)
        return img, x0,y0,x1,y1
        
    def get_heads_count(self):
        return len(self.bdot10k_cats_dict.keys())

    def get_label(self, df, idx):
        raise NotImplementedError
        
    @functools.lru_cache(maxsize=100)
    def load_bdot10k_for_powiat(self, powiat_id):
        df_to_merge: list = []
        for fn in self.shps:
            if powiat_id in fn:
                df = gpd.read_file(fn)
                df_to_merge.append(df)
        if len(df_to_merge) > 0:
            df_merged = pd.concat(df_to_merge)
            return df_merged
        else:
            return None
    
    def load_bdot10k_for_powiats(self, powiat_ids):
        df_to_merge: list = []
        for powiat_id in powiat_ids:
            df = self.load_bdot10k_for_powiat(powiat_id)
            df_to_merge.append(df)
        if len(df_to_merge)>0:
            df_merged = pd.concat(df_to_merge)
            return df_merged
        else:
            return None
    
    def powiat_ids_for_bbox(self, x0, y0, x1, y1):
        bbox = shapely.geometry.box(x0, y0, x1, y1)

        powiat_ids = []
        for p in self.powiaty.iloc:
            if p.geometry.contains(bbox) or bbox.contains(p.geometry):
                powiat_ids.append(p.JPT_KOD_JE)
        return powiat_ids
    
    def plot_sample(self, img, mask, show=False):
        raise NotImplementedError

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img, x0,y0,x1,y1 = self.get_image(idx)
        powiat_ids = self.powiat_ids_for_bbox(x0,y0,x1,y1)
        
        
        df = self.load_bdot10k_for_powiats(powiat_ids)
        if df is not None:
            df = df.cx[x0:x1, y0:y1]
            new_geoms = []
            for i in range(len(df)):
                s = df.iloc[i].geometry
                if 'MultiPolygon' in str(type(df.iloc[i].geometry)):
                    ps = []
                    for p in df.iloc[i].geometry:
                        xys = np.array([_ for _ in zip(*p.exterior.xy)])-(x0, y0)
                        xys[:,0]=xys[:,0]*img.shape[1]/(x1-x0)
                        xys[:,1]=xys[:,1]*img.shape[0]/(y1-y0)
                        ps.append(shapely.geometry.Polygon(xys))
                    s = shapely.geometry.MultiPolygon(ps)
                elif 'Polygon' in str(type(df.iloc[i].geometry)):
                    xys = np.array([_ for _ in zip(*df.iloc[i].geometry.exterior.xy)])-(x0, y0)
                    xys[:,0]=xys[:,0]*img.shape[1]/(x1-x0)
                    xys[:,1]=xys[:,1]*img.shape[0]/(y1-y0)
                    s = shapely.geometry.Polygon(xys)
                elif 'LineString' in str(type(df.iloc[i].geometry)):
                    xys = np.array(df.iloc[i].geometry.coords)-(x0, y0)
                    xys[:,0]=xys[:,0]*img.shape[1]/(x1-x0)
                    xys[:,1]=xys[:,1]*img.shape[0]/(y1-y0)
                    s = shapely.geometry.LineString(xys)
                elif 'MultiPoint' in str(type(df.iloc[i].geometry)):
                    ps = []
                    for p in df.iloc[i].geometry.geoms:
                        x = (p.x-x0)*img.shape[1]/(x1-x0)
                        y = (p.y-y0)*img.shape[0]/(y1-y0)
                        ps.append((x,y))
                    s = shapely.geometry.MultiPoint(ps)
                elif 'Point' in str(type(df.iloc[i].geometry)):
                    p = df.iloc[i].geometry
                    x = (p.x-x0)*img.shape[1]/(x1-x0)
                    y = (p.y-y0)*img.shape[0]/(y1-y0)
                    s = shapely.geometry.Point(x,y)
                else:
                    print(type(df.iloc[i].geometry))
                new_geoms.append(s)

            df.geometry = new_geoms
        else:
            df = None
                   
        label = self.get_label(df, idx)
        return img, label, os.path.basename(self.tiffs[idx])

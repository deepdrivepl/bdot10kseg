from rasterio.features import rasterize
import random
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
        self.tiffs = sorted(glob(tiff_dir+'/*.tif'))
        print(f'Found {len(self.tiffs)} tiff files in {self.tiff_dir}')
        self.powiaty = gpd.read_file(powiaty_shp_fname)
        print(f'Found {len(self.powiaty)} powiats')
        
        self.shps = sorted(glob(shp_dir+'/*.shp'))
        print(f'SHPs: {len(self.shps)}')
        
        self.bdot10k_df = pd.read_csv(bdot10k_cats_fname)
        self.bdot10k_cats = sorted(self.bdot10k_df.kod2.tolist())
        self.bdot10k_cats_dict = defaultdict(list)
        
        levels = {0:2, 1:4, 2:None}
        for k in self.bdot10k_cats:
            self.bdot10k_cats_dict[k[:levels[level]]].append(k)
            
        if classes is not None:
            self.bdot10k_cats_dict = {k:v for k,v in self.bdot10k_cats_dict.items() if k.startswith(tuple(classes))}
        print(f'Classes: {len(self.bdot10k_cats_dict)}')
        for k in self.bdot10k_cats_dict:
            print(f'{k}:', len(self.bdot10k_cats_dict[k]))
        
        self.transform = transform
        self.size = size
        self.level = level
        

    def __len__(self):
        return len(self.tiffs)

    def get_tile(self, raster):
        img = raster.read().transpose(1,2,0)
#         print(img.shape)
#         raster.transform * (0,0), raster.transform * (raster.width, raster.height)
        x0 = random.randint(0,img.shape[1]-self.size-1)
        y0 = random.randint(0,img.shape[0]-self.size-1)
        x1 = x0+self.size
        y1 = y0+self.size
        
#         print(x0,x1,y0,y1)
        _x0, _y0 = raster.transform * (x0,y0)
        _x1, _y1 = raster.transform * (x1,y1)
        
        _x0,_x1 = min(_x0,_x1), max(_x0,_x1)
        _y0,_y1 = min(_y0,_y1), max(_y0,_y1)
        
#         print(_x0,_y0, _x1,_y1)
#         print(x0,y0,x1,y1)
        
        return img[y0:y1,x0:x1], (_x0,_y0), (_x1,_y1) # crop coords in meters
    
    def get_id_by_code(self, code):
        return self.bdot10k_cats.index(code)
    
    def get_multiple_ids(self, code):
        return self.bdot10k_cats.index(code)
    
#     @functools.lru_cache(maxsize=10)
    def load_tiff(self, idx):
        tiff_fname = self.tiffs[idx]
#         print('loading:',tiff_fname)
        raster = rasterio.open(tiff_fname)
        return raster
    
    def get_image(self, idx):
        raster = self.load_tiff(idx)
        plt.figure(figsize=(20,20))
#         rasterio.plot.show(raster)
#         plt.show()
        img, (x0,y0),(x1,y1) = self.get_tile(raster)
        return img, x0,y0,x1,y1
        
    def get_heads_count(self):
        return len(self.bdot10k_cats_dict.keys())

    def get_label(self, df):
        mask = np.zeros((self.get_heads_count(), self.size, self.size), dtype='int')
        cat_names = list(self.bdot10k_cats_dict.keys())
        if df is not None:
            for code in df.X_KOD.unique():
                out_id = cat_names.index(code[:2])
        #         code_int = int(code[4:])
                code_int = self.get_id_by_code(code)
                m = code_int * rasterize(shapes=df[df.X_KOD==code].geometry.iloc,
                         out_shape=(self.size, self.size))[::-1]
                mask[out_id][m!=0] = m[m!=0]
        return mask
        
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
        cat_names = list(self.bdot10k_cats_dict.keys())
        fig = plt.figure(figsize=(20,10))
        plt.subplot(2,5,1)
        plt.imshow(img[:,:,::-1])
    #     plt.axis('off')
        for i,vmax in enumerate([len(_) for _ in self.bdot10k_cats_dict.values()]):
            plt.subplot(2,5,2+i)
    #         plt.imshow(mask[i],cmap='hsv')
            plt.imshow(mask[i],cmap='gist_ncar',vmin=0, vmax=vmax)
            plt.axis('off')
            plt.title(cat_names[i]+'\n'+ self.bdot10k_df[self.bdot10k_df.kod0==cat_names[i]].iloc[0].kategoria)
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

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img, x0,y0,x1,y1 = self.get_image(idx)
        
#         plt.figure(figsize=(10,10))
#         plt.imshow(img)
        
        powiat_ids = self.powiat_ids_for_bbox(x0,y0,x1,y1)
#         print(x0,y0,'-->',x1,y1,powiat_ids)
        
        
        df = self.load_bdot10k_for_powiats(powiat_ids)
        
#         df.geometry.plot(figsize=(10,10))
#         plt.show()
#         df.geometry.plot(figsize=(10,10))
#         plt.xlim(x0,x1)
#         plt.ylim(y0,y1)
#         plt.show()
#         print('all:',len(df))
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
    #                 print(df.iloc[i])
                new_geoms.append(s)

            df.geometry = new_geoms
        else:
            df = None
                   
        label = self.get_label(df)
        return img, label

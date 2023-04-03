import os
import math
import io

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BDOT10k(torch.utils.data.Dataset):
    
    def __init__(self,
                 tiff_list,
                 npy_dir,
                 bdot10k_cats_fname,
                 size=1024,
                 transform=None,
                 preprocessing=None
                ):
        
        self.tiffs = tiff_list
        
        self.npys = [x.replace(os.path.dirname(x), npy_dir).replace('.jpg', '.npy') for x in self.tiffs]
        self.npys = [x for x in self.npys if os.path.isfile(x)]
        
        self.tiffs = [x for x in self.tiffs if x.replace(os.path.dirname(x), npy_dir).replace('.jpg', '.npy') in self.npys]
        
        assert len(self.npys)==len(self.tiffs)
        print(f'IMGs {len(self.tiffs)}')
        print(f'LBLs: {len(self.npys)}')
        
        self.bdot10k_df = pd.read_csv(bdot10k_cats_fname)
        self.bdot10k_cats = sorted(list(set(self.bdot10k_df.kod0.tolist())))
        self.bdot10k_cats = [x for x in self.bdot10k_cats if x!="AD" and x!="PT"]
        self.ignore_idx = [0,4]
        print(f'Classes: {self.bdot10k_cats}')
        self.num_classes = len(self.bdot10k_cats)
        
        self.transform = transform
        self.preprocessing = preprocessing
        self.size = size
        
        
    def __len__(self):
        return len(self.tiffs)
    
    
    def get_sample(self, idx):
        npy_fname = self.npys[idx]
        try:
            lbl = np.load(npy_fname)
        except Exception as e:
            print(e)
            print('\n',npy_fname,'\n')
            raise
        lbl = lbl.transpose(1,2,0)
        for i in self.ignore_idx:
            lbl = np.delete(lbl, i, 2)
        
        tiff_fname = self.tiffs[idx]
        img = cv2.imread(tiff_fname)[:,:,::-1] # to RGB
        
        return img, lbl
    
    
    def __getitem__(self, idx):
        
        img, lbl = self.get_sample(idx)
        
        # apply augmentations
        if self.transform:
            sample = self.transform(image=img, mask=lbl)
            img, lbl = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=lbl)
            img, lbl = sample['image'], sample['mask']
            
        return img, lbl
    
    
    def plot_sample(self, img, mask, show=False):
        cat_names = self.bdot10k_cats
        
        n, m = 5, int(math.ceil((len(self.bdot10k_cats)+1)/5))
        fig = plt.figure(figsize=(n*5, m*5))
        plt.subplot(m,n,1)
        plt.imshow(img)
        
        for i in range(len(self.bdot10k_cats)):
            plt.subplot(m,n,2+i)
            plt.imshow(mask[:,:,i],cmap='gray',vmin=0, vmax=1)
            plt.axis('off')
            plt.title(cat_names[i]+'\n'+ self.bdot10k_df[self.bdot10k_df["kod0"]==cat_names[i]].iloc[0]["kategoria"])
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

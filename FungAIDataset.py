import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from skimage import io 
import mysql.connector
from get_annotation_pandas_df import get_annotation_pandas_df

class FungAIDataset(Dataset):
    def __init__(self, transform=None, limit=0):
        self.transform = transform
        self.annotations = get_annotation_pandas_df()
        self.annotations = self.annotations.dropna()
        self.annotations = self.annotations.reset_index(drop=True)
        
        if limit: self.annotations = self.annotations.iloc[0:limit]
        
    def __len__(self): return len(self.annotations)
    
    def __getitem__(self, index):
        img_path= Path(self.annotations["FrameIDPath"].iloc[19])
        image = io.imread(img_path)
        
        hyferanno = int(self.annotations["Hyfer"].iloc[index])
        y_label = torch.tensor(0 if hyferanno == 0 else 1)
        
        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)

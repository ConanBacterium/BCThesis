import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path 
from PIL import Image
# import mysql.connector
from get_annotation_pandas_df import get_annotation_pandas_df

class FungAIDataset(Dataset):
    def __init__(self, transform=None, limit=0, balanced=False):
        self.transform = transform
        self.annotations = get_annotation_pandas_df()
        self.annotations = self.annotations.dropna()
        self.annotations = self.annotations.reset_index(drop=True)
        
        if balanced: 
            self._balance_annotations(666)
            sanity_check = self.annotations['Hyfer'][self.annotations['Hyfer'] > 0].count() == self.annotations['Hyfer'][self.annotations['Hyfer'] == 0].count()
            print(f"balanced: {sanity_check}")
        if limit: self.annotations = self.annotations.iloc[0:limit]
            
    def _balance_annotations(self, randomSeed):
        num_nonzeros = self.annotations['Hyfer'][self.annotations['Hyfer'] > 0].count()
        num_zeros = self.annotations['Hyfer'][self.annotations['Hyfer'] == 0].count()
        while num_nonzeros != num_zeros:
            num_to_remove = num_zeros - num_nonzeros
            remove_indices = self.annotations[self.annotations['Hyfer'] == 0].sample(n=num_to_remove, random_state=randomSeed).index
            self.annotations = self.annotations.drop(remove_indices)
            num_nonzeros = self.annotations['Hyfer'][self.annotations['Hyfer'] > 0].count()
            num_zeros = self.annotations['Hyfer'][self.annotations['Hyfer'] == 0].count()
        
    def __len__(self): return len(self.annotations)
    
    def __getitem__(self, index):
        img_path= Path(self.annotations["FrameIDPath"].iloc[19])
        image = Image.open(img_path)
        
        hyferanno = int(self.annotations["Hyfer"].iloc[index])
        y_label = torch.tensor(0 if hyferanno == 0 else 1)
        
        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)

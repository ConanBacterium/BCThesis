import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path 
from PIL import Image
# import mysql.connector
from get_annotation_pandas_df import get_annotation_pandas_df

class FungAIDataset(Dataset):
    def __init__(self, annotations_pd_df=None, transform=None, limit=0, balanced=False):
        self.transform = transform
        if annotations_pd_df is None: 
            self.annotations = get_annotation_pandas_df()
            self.annotations = self.annotations.dropna()
            self.annotations = self.annotations.reset_index(drop=True)
        else: self.annotations = annotations_pd_df
        
        if balanced: self._balance_annotations(666)
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
        img_path= Path(self.annotations["FrameIDPath"].iloc[index])
        image = Image.open(img_path)
        
        hyferanno = int(self.annotations["Hyfer"].iloc[index])
        y_label = torch.tensor(0 if hyferanno == 0 else 1)
        
        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)
    
# if given valsize = 0 it will only return train and test Pytorch datasets. 
def getFungAIDatasetSplits(valsize, testsize, trainsize=None, train_transform=None, val_test_transform=None, limit=0, balanced=False, randomSeed=666):
    #### TODO !!! SHUFFLE?!
    annotations = get_annotation_pandas_df()
    annotations = annotations.dropna()
    annotations = annotations.reset_index(drop=True)
    
    if balanced: 
        num_nonzeros = annotations['Hyfer'][annotations['Hyfer'] > 0].count()
        num_zeros = annotations['Hyfer'][annotations['Hyfer'] == 0].count()
        while num_nonzeros != num_zeros:
            num_to_remove = num_zeros - num_nonzeros
            remove_indices = annotations[annotations['Hyfer'] == 0].sample(n=num_to_remove, random_state=randomSeed).index
            annotations = annotations.drop(remove_indices)
            num_nonzeros = annotations['Hyfer'][annotations['Hyfer'] > 0].count()
            num_zeros = annotations['Hyfer'][annotations['Hyfer'] == 0].count()
            
    if limit: annotations = annotations.iloc[0:limit]
        
    if trainsize is None: trainsize = len(annotations) - valsize - testsize
        
    trainset = annotations.iloc[0:trainsize].reset_index(drop=True)
    if valsize: 
        valset = annotations.iloc[trainsize:trainsize+valsize].reset_index(drop=True)
        testset = annotations.iloc[trainsize+valsize:trainsize+valsize+testsize].reset_index(drop=True)
        
        return FungAIDataset(trainset, train_transform), FungAIDataset(valset, val_test_transform), FungAIDataset(testset, val_test_transform)
    else: 
        testset = annotations.iloc[trainsize:trainsize+testsize].reset_index(drop=True)
        return FungAIDataset(trainset, val_test_transform), FungAIDataset(testset, val_test_transform)
        
    

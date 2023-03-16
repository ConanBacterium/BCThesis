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
    
# this balances every split. The splits don't overlap. 
def create_balanced_splits(annotations, trainsize, valsize, testsize):
    # Select samples with positive and negative labels
    pos_samples = annotations[annotations['Hyfer'] > 0]
    neg_samples = annotations[annotations['Hyfer'] == 0]
    
    # Shuffle the samples
#     pos_samples = pos_samples.sample(frac=1)
#     neg_samples = neg_samples.sample(frac=1)
    
    # Calculate the number of samples for each set
    n_pos = len(pos_samples)
    n_neg = len(neg_samples)
    n_total = n_pos + n_neg
    
#     n_train = int(n_total * trainsize)
#     n_val = int(n_total * valsize)
#     n_test = int(n_total * testsize)
    n_train = trainsize
    n_val = valsize
    n_test = testsize
    
    # Calculate the number of positive and negative samples for each set
    n_pos_train = int(n_train * (n_pos / n_total))
    n_neg_train = n_train - n_pos_train
    
    n_pos_val = int(n_val * (n_pos / n_total))
    n_neg_val = n_val - n_pos_val
    
    n_pos_test = int(n_test * (n_pos / n_total))
    n_neg_test = n_test - n_pos_test
    
    # Sample randomly from positive and negative samples for each set
    pos_samples_train = pos_samples.iloc[:n_pos_train]
    neg_samples_train = neg_samples.iloc[:n_neg_train]
    train_df = pd.concat([pos_samples_train, neg_samples_train])
    
    pos_samples_val = pos_samples.iloc[n_pos_train:n_pos_train+n_pos_val]
    neg_samples_val = neg_samples.iloc[n_neg_train:n_neg_train+n_neg_val]
    val_df = pd.concat([pos_samples_val, neg_samples_val])
    
    pos_samples_test = pos_samples.iloc[n_pos_train+n_pos_val:n_pos_train+n_pos_val+n_pos_test]
    neg_samples_test = neg_samples.iloc[n_neg_train+n_neg_val:n_neg_train+n_neg_val+n_neg_test]
    test_df = pd.concat([pos_samples_test, neg_samples_test])
    
    return train_df, val_df, test_df

# if given valsize = 0 it will only return train and test Pytorch datasets. 
def getFungAIDatasetSplits(valsize, testsize, trainsize=None, train_transform=None, val_test_transform=None, limit=0, balanced=False, randomSeed=666):
    
    
    #### TEST SÆT SKAL ALTID VÆRE BALANCERET !!! 
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
        
    # TODO !! MAKE SHURE trainsize - valsize - testsize  => 0
    assert trainsize - valsize - testsize >= 0
    
    train_df, val_df, test_df = create_balanced_splits(annotations, trainsize, valsize, testsize)
    
    if valsize: 
        return FungAIDataset(train_df, train_transform), FungAIDataset(val_df, val_test_transform), FungAIDataset(test_df, val_test_transform)
    else: 
        return FungAIDataset(train_df, train_transform), FungAIDataset(test_df, val_test_transform)
        
#     trainset = annotations.iloc[0:trainsize].reset_index(drop=True)
#     if valsize: 
#         valset = annotations.iloc[trainsize:trainsize+valsize].reset_index(drop=True)
#         testset = annotations.iloc[trainsize+valsize:trainsize+valsize+testsize].reset_index(drop=True)
        
#         return FungAIDataset(trainset, train_transform), FungAIDataset(valset, val_test_transform), FungAIDataset(testset, val_test_transform)
#     else: 
#         testset = annotations.iloc[trainsize:trainsize+testsize].reset_index(drop=True)
#         return FungAIDataset(trainset, val_test_transform), FungAIDataset(testset, val_test_transform)
        
    

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io 

class FungAIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(os.path.join(root_dir, 'data.csv'))
        self.data = self.data.dropna()
        self.data = self.data.reset_index(drop=True)
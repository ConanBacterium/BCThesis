from generateAugmentedImgTensors import generateAugmentedImgTensors 
import os
import torch

def returnTifsInDir(dir_path):
    tif_files = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.tif'):
            file_path = os.path.join(dir_path, file_name)
            tif_files.append(file_path)
    return tif_files

factors = torch.linspace(0.39, 1.46, 10).tolist() ##### would be better if it wasn't linear but instead inverted gaussian with mean 1 so least likely around 1... Since distance matters more near the boundaries. 
            
normals = returnTifsInDir("data\\ICIAR2018_BACH_Challenge\\Photos\\Normal")
generateAugmentedImgTensors(normals, True, True, "data/ICIAR2018_BACH_Challenge/Photos/Normal_brightness_aug_prepost_norm", factors)

benign = returnTifsInDir("data\\ICIAR2018_BACH_Challenge\\Photos\\Benign")
generateAugmentedImgTensors(benign, True, True, "data/ICIAR2018_BACH_Challenge/Photos/Benign_brightness_aug_prepost_norm", factors)

insitu = returnTifsInDir("data\\ICIAR2018_BACH_Challenge\\Photos\\InSitu")
generateAugmentedImgTensors(insitu, True, True, "data/ICIAR2018_BACH_Challenge/Photos/InSitu_brightness_aug_prepost_norm", factors)

invasive = returnTifsInDir("data\\ICIAR2018_BACH_Challenge\\Photos\\Invasive")
generateAugmentedImgTensors(invasive, True, True, "data/ICIAR2018_BACH_Challenge/Photos/Invasive_brightness_aug_prepost_norm", factors)
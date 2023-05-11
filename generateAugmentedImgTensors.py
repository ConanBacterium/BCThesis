from PIL import Image
import os
from pathlib import Path
import torch
import torch.nn as nn # all neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # all optimization algorithms, SGD, Adam, etc.
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F # all functions that don't have any parameters, relu, sigmoid, softmax, etc.
from torch.utils.data import DataLoader # gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets # has standard datasets we can import in a nice way
import torchvision.transforms as transforms # transform images, videos, etc.
import torchvision.models as models

def returnTifsInDir(dir_path):
    tif_files = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.tif'):
            file_path = os.path.join(dir_path, file_name)
            tif_files.append(file_path)
    return tif_files

normalisation_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def tensorToPil(pilimg):
    img = pilimg.numpy()
#     img = (img - img.min()) / (img.max() - img.min()) * 255 # this is the magic sauce. Looks like it makes the colors a little blue??
    img = img.astype(np.uint8)
    img = np.transpose(img, (1,2,0))
    img = Image.fromarray(img)
    return img 

def generateAugmentedImgTensors(tif_paths, pre_normalize, post_normalize, targetdir, factors):
    if not os.path.exists(targetdir): os.makedirs(targetdir)
    
    savepaths = []

    for i, tif_path in enumerate(tif_paths):
        img = Image.open(tif_path)
        
        if pre_normalize:
            img = normalisation_transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        filename = Path(tif_path).name.split(".")[0]

        for factor in factors: 

            brightened_img = img * factor
            # Clip values to keep them in the valid range [0, 1]
            brightened_img = torch.clamp(brightened_img, 0, 1)
            
            if post_normalize: 
#                 brightened_img = normalisation_transform(brightened_img).numpy().transpose((1,2,0))
                brightened_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(brightened_img)

            
            savepath = f'{targetdir}/{filename}_brightened_{factor}.pt'
            torch.save(brightened_img, savepath)
            savepaths.append(savepath)
    return savepaths
            

# no_norm_factors = torch.linspace(0.2, 1.84, 15).tolist() ##### would be better if it wasn't linear but instead concave, so least likely around 1... Since distance matters more near the boundaries. 
            
# normals = returnTifsInDir("Normal")
# generatePlotOfAugmentationsBrightness(normals, True, True, "Normal_brightness_aug_prepost_norm", no_norm_factors)

# benign = returnTifsInDir("Benign")
# generatePlotOfAugmentationsBrightness(benign, True, True, "Benign_brightness_aug_prepost_norm", no_norm_factors)

# insitu = returnTifsInDir("InSitu")
# generatePlotOfAugmentationsBrightness(insitu, True, True, "InSitu_brightness_aug_prepost_norm", no_norm_factors)

# invasive = returnTifsInDir("Invasive")
# generatePlotOfAugmentationsBrightness(invasive, True, True, "Invasive_brightness_prepost_norm", no_norm_factors)
from get_descriptor_from_imgtensor import get_descriptor_from_imgtensor
import os
from tqdm import tqdm
from pathlib import Path
import torch
from torchvision.transforms import functional as TF
import torchvision.models as models

def returnPtFilesInDir(dir_path):
    pt_files = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.pt'):
            file_path = os.path.join(dir_path, file_name)
            pt_files.append(file_path)
    return pt_files



efficientnet = models.efficientnet_b7(pretrained=True)
efficientnet = torch.nn.Sequential(*(list(efficientnet.children())[:-1])) # remove last layer 
efficientnet.eval()
efficientnet.to("cpu")

normal_dir = "data/ICIAR2018_BACH_Challenge/Photos/Normal_brightness_aug_prepost_norm"
benign_dir = "data/ICIAR2018_BACH_Challenge/Photos/Benign_brightness_aug_prepost_norm"
insitu_dir = "data/ICIAR2018_BACH_Challenge/Photos/InSitu_brightness_aug_prepost_norm"
invasive_dir = "data/ICIAR2018_BACH_Challenge/Photos/Invasive_brightness_aug_prepost_norm"

normal_targetdir = "data/ICIAR2018_BACH_Challenge/Photos/Normal_descriptors"
benign_targetdir = "data/ICIAR2018_BACH_Challenge/Photos/Benign_descriptors"
insitu_targetdir = "data/ICIAR2018_BACH_Challenge/Photos/InSitu_descriptors"
invasive_targetdir = "data/ICIAR2018_BACH_Challenge/Photos/Invasive_descriptors"

normal_imgtensors = returnPtFilesInDir(normal_dir)
benign_imgtensors = returnPtFilesInDir(benign_dir)
insitu_imgtensors = returnPtFilesInDir(insitu_dir)
invasive_imgtensors = returnPtFilesInDir(invasive_dir)

for imgtensor_list, targetdir in [(normal_imgtensors, normal_targetdir), (benign_imgtensors, benign_targetdir), (insitu_imgtensors, insitu_targetdir), (invasive_imgtensors, invasive_targetdir)]:
    if not os.path.exists(targetdir): os.makedirs(targetdir)
    for imgtensor_path in tqdm(imgtensor_list):
        imgtensor_name = Path(imgtensor_path).name
        imgtensor = torch.load(imgtensor_path)
        descriptor = get_descriptor_from_imgtensor(imgtensor, 800, 1300, 7, 3, efficientnet)

        torch.save(descriptor, f"{targetdir}/{imgtensor_name}")

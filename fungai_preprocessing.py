from pointwise_crops_w_touchcount import pointwise_crops_w_touchcount
from generateAugmentedImgTensors import generateAugmentedImgTensors
from get_descriptor_from_imgtensor import get_descriptor_from_imgtensor
import torch 
import torchvision.models as models
device = torch.device("cpu")
from pathlib import Path
import os
import time

preprocessing_dir = Path("data/fungai_preprocessing")
crops_dir = preprocessing_dir / "crops"
augmented_tensor_crops_dir = preprocessing_dir / "augmented_tensor_crops"
descriptors_dir = preprocessing_dir / "descriptors"

if not os.path.exists(preprocessing_dir): os.makedirs(preprocessing_dir)
if not os.path.exists(crops_dir): os.makedirs(crops_dir)
if not os.path.exists(augmented_tensor_crops_dir): os.makedirs(augmented_tensor_crops_dir)
if not os.path.exists(descriptors_dir): os.makedirs(descriptors_dir)

def get_sagsinfo(path_of_full_png):
    string = Path(path_of_full_png).name
    if string[:2] == "A_": string = string[2:]
    
    sagsinfo = string.split(".")[0]
    return sagsinfo

######## TODO #########
######## TODO #########
# CHANGE SO BRIGHTNESS AUGMENTATIONS ARE DONE AS THE PNG IS CUT UP!!! TO BEGIN WITH. That way don't need to load a many times over
######## TODO #########
######## TODO #########
def fungai_preprocess_wholeimg(pngpath, maskpath, postnorm, prenorm, brightness_factors):
    # cut up into appropriate iciar size and save the touchcount in name
    sagsinfo = get_sagsinfo(pngpath)
    crops_savedir = crops_dir / sagsinfo 
    if not os.path.exists(crops_savedir): os.makedirs(crops_savedir)
    
    # print("sagsinfo: ", sagsinfo)
    # print("crops_savedir: ", crops_savedir)
    img_crop_paths, mask_crop_paths = pointwise_crops_w_touchcount(pngpath, maskpath, margincount_logfile_path = "data/fungai_preprocessing/margincounts.csv", savedir=crops_savedir, cropsize=(2048, 1536), overlap=400)
    
    # do brightness augmentations 
    augmented_img_tensors_targetdir = augmented_tensor_crops_dir / sagsinfo
    if not os.path.exists(augmented_img_tensors_targetdir): os.makedirs(augmented_img_tensors_targetdir)

    # print("augmented_img_tensors_targetdir: ", augmented_img_tensors_targetdir)
    augmented_img_tensors_paths = generateAugmentedImgTensors(img_crop_paths, prenorm, postnorm, augmented_img_tensors_targetdir, brightness_factors)
    
    
    efficientnet = models.efficientnet_b7(pretrained=True)
    efficientnet = torch.nn.Sequential(*(list(efficientnet.children())[:-1])) # remove last layer 
    efficientnet.eval()
    efficientnet.to(device)
    
    # extract crops, encode and 3normpooling to save descriptor
    for img_tensor_path in augmented_img_tensors_paths:
        img_tensor_name = Path(img_tensor_path).name
        img_tensor = torch.load(img_tensor_path)
        pooled_descriptor = get_descriptor_from_imgtensor(img_tensor, 800, 1300, 7, 3, efficientnet)
        torch.save(pooled_descriptor, descriptors_dir / img_tensor_name)
    
    
directory = Path("data/FungAI")
mask_paths = []
png_paths = []

for file_path in directory.glob("*"):
    if file_path.name.startswith("A_"):
        mask_paths.append(file_path)
    else:
        png_paths.append(file_path)

mask_paths.sort()
png_paths.sort()

factors = torch.linspace(0.39, 1.64, 10).tolist()

for i, (mask_path, png_path) in enumerate(zip(mask_paths, png_paths)):
    print(f"{i}/{len(mask_paths)}")

    start_time = time.time()
    
    fungai_preprocess_wholeimg(png_path, mask_path, True, True, factors)

    end_time = time.time()

    print("Elapsed time: {:.2f} seconds".format(end_time - start_time))
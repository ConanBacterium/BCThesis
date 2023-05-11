# for each augmented image, extract 20 crops (800x800 and 1300x1300) and encode them using effnet, densenet and resnet. 


import os
import torch
from torchvision.transforms import functional as TF
import torchvision.models as models
import time
from pathlib import Path
import sys
import shutil
import os

device = torch.device("cuda")

def geometric_mean(tensor, dim, power):
    pow_tensor = torch.pow(tensor.abs(), power)
    log_pow_tensor = torch.log(pow_tensor)
    mean_log_pow_tensor = log_pow_tensor.mean(dim)
    return torch.exp(mean_log_pow_tensor / power)

# Nonbatch 7.306056976318359 seconds to execute.
def random_crops_and_encode_nonbatch(img_tensor, cropsize1, cropsize2, cropcount1, cropcount2, efficientnet):
    def random_crop(tensor, crop_size):
        _, h, w = tensor.shape
        top = torch.randint(0, h - crop_size + 1, (1,)).item()
        left = torch.randint(0, w - crop_size + 1, (1,)).item()
        return tensor[:, top:top + crop_size, left:left + crop_size]

    crops = []
    for _ in range(cropcount1):
        crops.append(random_crop(img_tensor, cropsize1))

    for _ in range(cropcount2):
        crops.append(random_crop(img_tensor, cropsize2))
    
    encodings = []
    with torch.no_grad():
        for crop in crops:
            input_tensor = crop.unsqueeze(0)
            input_tensor = input_tensor.to(device)
            efficientnet_encoding = efficientnet(input_tensor)

            encodings.extend(efficientnet_encoding)

    encodings_tensor = torch.stack(encodings)
    pooled_descriptor = torch.pow(torch.pow(encodings_tensor, 3).mean(dim=0), 1/3)

    return pooled_descriptor

# Batch 2.05796217918396 seconds to execute.
def random_crops_and_encode_batch(img_tensor, cropsize1, cropsize2, cropcount1, cropcount2, efficientnet):
    img_tensor = img_tensor.to(device)
    def random_crop(tensor, crop_size):
        _, h, w = tensor.shape
        top = torch.randint(0, h - crop_size + 1, (1,)).item()
        left = torch.randint(0, w - crop_size + 1, (1,)).item()
        cropped_tensor = tensor[:, top:top + crop_size, left:left + crop_size]
        cropped_tensor = cropped_tensor.to(device)
        return cropped_tensor

    crops1 = []
    for _ in range(cropcount1):
        crops1.append(random_crop(img_tensor, cropsize1))

    crops2 = []
    for _ in range(cropcount2):
        crops2.append(random_crop(img_tensor, cropsize2))
    
    encodings = []

    batch_size = 20

    with torch.no_grad():
        for i in range(0, len(crops1), batch_size):
            batch_crops = crops1[i:i + batch_size]
            input_tensor = torch.stack(batch_crops)
            input_tensor = input_tensor.to(device)
            efficientnet_encoding = efficientnet(input_tensor)
            encodings.extend(efficientnet_encoding)
         
        for i in range(0, len(crops2), batch_size):
            batch_crops = crops2[i:i + batch_size]
            input_tensor = torch.stack(batch_crops)
            input_tensor = input_tensor.to(device)
            efficientnet_encoding = efficientnet(input_tensor)
            encodings.extend(efficientnet_encoding)

    encodings_tensor = torch.stack(encodings)

    if torch.isnan(encodings_tensor).any():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("There are NaN values in encodings_tensor")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    #pooled_descriptor = torch.pow(torch.pow(encodings_tensor, 3).mean(dim=0), 1/3)
#     epsilon = 1e-8
#     pooled_descriptor = torch.pow(torch.pow(encodings_tensor, 3).mean(dim=0) + epsilon, 1/3)
    pooled_descriptor = geometric_mean(encodings_tensor, 0, 3)
    
    if torch.isnan(pooled_descriptor).any():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("There are NaN values in pooled_descriptor")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    return pooled_descriptor


efficientnet = models.efficientnet_b7(pretrained=True)
efficientnet = torch.nn.Sequential(*(list(efficientnet.children())[:-1])) # remove last layer 
efficientnet.eval()
efficientnet.to(device)

def returnPngsInDir(dir_path):
    tif_files = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.pt'):
            file_path = os.path.join(dir_path, file_name)
            tif_files.append(file_path)
    return tif_files

normals = returnPngsInDir("Normal_brightness_aug_prepost_norm")
insitu = returnPngsInDir("InSitu_brightness_aug_prepost_norm")
invasive = returnPngsInDir("Invasive_brightness_prepost_norm")
benign = returnPngsInDir("Benign_brightness_aug_prepost_norm")

target_parentdir = Path("A1_brightness_aug_prepost_norm_3norm_effnet_descriptors")
if not os.path.exists(target_parentdir):
    os.makedirs(target_parentdir)
    subdirs = ["Normal", "InSitu", "Invasive", "Benign"]
    for subdir in subdirs: os.makedirs(os.path.join(target_parentdir, subdir))
    current_file_path = os.path.abspath(__file__)
    shutil.copy2(current_file_path, target_parentdir / "GENERATOR_SCRIPT.py")
else: 
    print(f"{target_parentdir} already exists")
    sys.exit()

for i, tensorpaths in enumerate(zip(normals, insitu, invasive, benign)):
    start_time = time.time()
    for j, tensorpath in enumerate(tensorpaths):
        tensorpath = Path(tensorpath)
        tensorname = tensorpath.name.split(".")[0] + tensorpath.name.split(".")[1] # there are two dots, unfortunately... 
        if j == 0: targetdir = target_parentdir / "Normal"
        elif j == 1: targetdir = target_parentdir / "InSitu"
        elif j == 2: targetdir = target_parentdir / "Invasive"
        elif j == 3: targetdir = target_parentdir / "Benign"
            
        imgtensor = torch.load(tensorpath)
        pooled_descriptor = random_crops_and_encode_batch(imgtensor, 800, 1300, 13, 7, efficientnet)
        pooled_descriptor = pooled_descriptor.to("cpu")
        torch.save(pooled_descriptor, f"{targetdir}/{tensorname}")
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"{i+1}/{len(normals)}: {elapsed_time} seconds to execute.")

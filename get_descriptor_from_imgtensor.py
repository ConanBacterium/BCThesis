import os
import torch
from torchvision.transforms import functional as TF
import torchvision.models as models
import time
from pathlib import Path
import sys
import shutil
import os

if torch.cuda.is_available():   device = torch.device("cuda")  # Use CUDA device
else:    device = torch.device("cpu")   # Use CPU

def geometric_mean(tensor, dim, power):
    pow_tensor = torch.pow(tensor.abs(), power)
    log_pow_tensor = torch.log(pow_tensor)
    mean_log_pow_tensor = log_pow_tensor.mean(dim)
    return torch.exp(mean_log_pow_tensor / power)

# Batch 2.05796217918396 seconds to execute.
# extracting random crops and encoding with efficientnet and 3-norm pooling
def get_descriptor_from_imgtensor(img_tensor, cropsize1, cropsize2, cropcount1, cropcount2, efficientnet):
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
        # torch.save(crops1[-1], f"data/testing_random_crops/random_crop_{_}.pt")

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
    
    pooled_descriptor = geometric_mean(encodings_tensor, 0, 3)
    return pooled_descriptor

def get_descriptor_from_imgtensor__chunks_instead_of_random_crops(img_tensor, efficientnet):
    img_tensor = img_tensor.to(device) 
    def get_chunks(image_tensor, num_crops_vertical, num_crops_horizontal):

        crops = torch.chunk(image_tensor, num_crops_vertical, dim=1) # dims: 0->channels, 1->horizontal, 2->vertical
        cropped_images = [torch.chunk(crop, num_crops_horizontal, dim=2) for crop in crops]
        cropped_images = [item for sublist in cropped_images for item in sublist]

        return cropped_images
    def pad_(cropped_images, max_height, max_width):
        padded_images = []
        for crop in cropped_images:
            pad_height = max_height - crop.size(1)
            pad_width = max_width - crop.size(2)
            padded_crop = torch.nn.functional.pad(crop, (0, pad_width, 0, pad_height))
            padded_images.append(padded_crop)
        return padded_images
    
    crops_4 = get_chunks(img_tensor, 2, 2)
    crops_9 = get_chunks(img_tensor, 3, 3)

    # start_time = time.time()
    max_height_4 = max([crop.size(1) for crop in crops_4])
    max_width_4 = max([crop.size(2) for crop in crops_4])
    max_height_9 = max([crop.size(1) for crop in crops_9])
    max_width_9 = max([crop.size(2) for crop in crops_9])
    max_height = max([max_height_4, max_height_9])
    max_width = max([max_width_4, max_width_9])
    crops_4 = pad_(crops_4, max_height, max_width)
    crops_9 = pad_(crops_9, max_height, max_width)
    # elapsed_time = time.time() - start_time
    # print("Padding took:", elapsed_time, "seconds") ### takes 0.012269735336303711, so no problemo
    
    encodings = []

    batch_size = 20

    with torch.no_grad():
        for i in range(0, len(crops_4), batch_size):
            batch_crops = crops_4[i:i + batch_size]
            input_tensor = torch.stack(batch_crops)
            input_tensor = input_tensor.to(device)
            efficientnet_encoding = efficientnet(input_tensor)
            encodings.extend(efficientnet_encoding)
         
        for i in range(0, len(crops_9), batch_size):
            batch_crops = crops_9[i:i + batch_size]
            input_tensor = torch.stack(batch_crops)
            input_tensor = input_tensor.to(device)
            efficientnet_encoding = efficientnet(input_tensor)
            encodings.extend(efficientnet_encoding)

    encodings_tensor = torch.stack(encodings)
    
    pooled_descriptor = geometric_mean(encodings_tensor, 0, 3)
    return pooled_descriptor


# efficientnet = models.efficientnet_b7(pretrained=True)
# efficientnet = torch.nn.Sequential(*(list(efficientnet.children())[:-1])) # remove last layer 
# efficientnet.eval()
# efficientnet.to(device)

# def returnPngsInDir(dir_path):
#     tif_files = []
#     for file_name in os.listdir(dir_path):
#         if file_name.endswith('.pt'):
#             file_path = os.path.join(dir_path, file_name)
#             tif_files.append(file_path)
#     return tif_files

# normals = returnPngsInDir("Normal_brightness_aug_prepost_norm")
# insitu = returnPngsInDir("InSitu_brightness_aug_prepost_norm")
# invasive = returnPngsInDir("Invasive_brightness_prepost_norm")
# benign = returnPngsInDir("Benign_brightness_aug_prepost_norm")

# target_parentdir = Path("A1_brightness_aug_prepost_norm_3norm_effnet_descriptors")
# if not os.path.exists(target_parentdir):
#     os.makedirs(target_parentdir)
#     subdirs = ["Normal", "InSitu", "Invasive", "Benign"]
#     for subdir in subdirs: os.makedirs(os.path.join(target_parentdir, subdir))
#     current_file_path = os.path.abspath(__file__)
#     shutil.copy2(current_file_path, target_parentdir / "GENERATOR_SCRIPT.py")
# else: 
#     print(f"{target_parentdir} already exists")
#     sys.exit()

# for i, tensorpaths in enumerate(zip(normals, insitu, invasive, benign)):
#     start_time = time.time()
#     for j, tensorpath in enumerate(tensorpaths):
#         tensorpath = Path(tensorpath)
#         tensorname = tensorpath.name.split(".")[0] + tensorpath.name.split(".")[1] # there are two dots, unfortunately... 
#         if j == 0: targetdir = target_parentdir / "Normal"
#         elif j == 1: targetdir = target_parentdir / "InSitu"
#         elif j == 2: targetdir = target_parentdir / "Invasive"
#         elif j == 3: targetdir = target_parentdir / "Benign"
            
#         imgtensor = torch.load(tensorpath)
#         pooled_descriptor = random_crops_and_encode_batch(imgtensor, 800, 1300, 13, 7, efficientnet)
#         pooled_descriptor = pooled_descriptor.to("cpu")
#         torch.save(pooled_descriptor, f"{targetdir}/{tensorname}")
#     end_time = time.time()
    
#     elapsed_time = end_time - start_time
#     print(f"{i+1}/{len(normals)}: {elapsed_time} seconds to execute.")
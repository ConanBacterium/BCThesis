from pathlib import Path

path_descriptors_randomized = Path("data/fungai_preprocessing/descriptors_randomized")  
path_descriptors_chunks = Path("data/fungai_preprocessing/descriptors_chunks")  

randomized_descriptors_fps = list(path_descriptors_randomized.glob('*.pt'))  
print(len(randomized_descriptors_fps))

chunked_descriptors_fps = list(path_descriptors_chunks.glob('*.pt'))  
print(len(chunked_descriptors_fps))

randomized_names = [e.name for e in randomized_descriptors_fps]
chunked_names = [e.name for e in chunked_descriptors_fps]

assert randomized_names == chunked_names
print("randomized and chunked have same filenames in same order")

descriptor_filenames = chunked_names 

# there are 10 brightness augmentations per crop. The test set should only include the brightness closest to 1, and not
# the other augmentations. So sort the crops

def get_sublists_of_10(lst):
    n = len(lst)
    slices = []
    for i in range(0, n, 10):
        slices.append(lst[i:i+10])
    return slices

descriptor_filenames.sort()
descriptor_cropnames = get_sublists_of_10(descriptor_filenames)

# shuffle 
import random
random.seed(666)
random.shuffle(descriptor_cropnames)

brightness_factor_closest_to_1 = "1.0844444036483765"

a = int(len(descriptor_cropnames) * 0.25)

test_cropnames = descriptor_cropnames[:a]
train_cropnames = descriptor_cropnames[a:]

print(len(test_cropnames))
print(len(train_cropnames))

import numpy as np 
import matplotlib.pyplot as plt
import torch
from pathlib import Path

def get_touchcount_from_descriptorname(descriptorname):
    return int(descriptorname.split("_")[6])

def getTrainTestTensorsAndLabels(positive_threshold, descriptor_parentdir, train_cropnames, test_cropnames):
    #Capture training data and labels into respective lists
    train_images = []
    train_labels = [] 
    test_images = []
    test_labels = []

    for crop_names in train_cropnames: 
        for cropname in crop_names:
            descriptor_tensor = torch.load(descriptor_parentdir/cropname).flatten()
            train_images.append(descriptor_tensor)
            touchcount = get_touchcount_from_descriptorname(cropname)
            if touchcount >= positive_threshold: 
                train_labels.append(1)
            # DONT APPEND TO NEGATIVES IF LOWER THRESHOLD UNLESS THRESHOLD IS 0 
            elif touchcount == 0: 
                train_labels.append(0)
            #else: train_labels.append(0) 

    for crop_names in test_cropnames: 
        for cropname in crop_names:
            if brightness_factor_closest_to_1 not in cropname: continue # this is the brightness value closest to 1... Not perfect, but I'm an idiot
            descriptor_tensor = torch.load(descriptor_parentdir/cropname).flatten()
            test_images.append(descriptor_tensor)
            touchcount = get_touchcount_from_descriptorname(cropname)
            if touchcount >= positive_threshold: 
                train_labels.append(1)
            # DONT APPEND TO NEGATIVES IF LOWER THRESHOLD UNLESS THRESHOLD IS 0 
            elif touchcount == 0: 
                train_labels.append(0)
            #else: train_labels.append(0) 

    #Convert lists to arrays        
    train_images = torch.stack(train_images, dim=0)
    train_labels = torch.tensor(train_labels)
    
    assert train_labels.sum() > 0

    test_images = torch.stack(test_images, dim=0)
    test_labels = torch.tensor(test_labels)
    
    assert test_labels.sum() > 0
    
    return train_images, train_labels, test_images, test_labels

def train_and_get_test_metrics_of_threshold(positive_threshold, train_crops, test_crops, descriptor_parentdir):
    train_images, train_labels, test_images, test_labels = getTrainTestTensorsAndLabels(positive_threshold, descriptor_parentdir, train_crops, test_crops)
    model = train_and_get_test_metrics(train_images, train_labels, test_images, test_labels)
    return model

import pickle 

def save_pickle_of_dict_of_thresholded_train_and_test(descriptor_parentdir, pickle_name_prefix, train_cropnames, test_cropnames):
    for threshold in range(10):
        train_images, train_labels, test_images, test_labels = getTrainTestTensorsAndLabels(threshold, descriptor_parentdir, train_cropnames, test_cropnames)
        d = {"train_images": train_images, "train_labels": train_labels, "test_images": test_images, "test_labels": test_labels}
        savepath = Path(f"data/fungai_preprocessing/{pickle_name_prefix}_threshold_{threshold}")
        with open(savepath, "wb") as f:
            pickle.dump(d, f)

save_pickle_of_dict_of_thresholded_train_and_test(path_descriptors_chunks, "chunked", train_cropnames, test_cropnames)
save_pickle_of_dict_of_thresholded_train_and_test(path_descriptors_chunks, "randomized", train_cropnames, test_cropnames)
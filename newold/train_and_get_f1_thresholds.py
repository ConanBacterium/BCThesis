# THIS IS PORTED FROM "gradient boosted tree on NEW BALANCED FUNGAI ICIAR.ipynb" BECAUSE DESKTOP PC DOESNT HAVE PROPER ENVIRONMENT FOR JUPYTER !!! 

import torch
import pickle as pkl
from pathlib import Path

with open(("data/descriptors_dict_threshold_1_through_10.pkl"), 'rb') as f:
    # Load the object from the pickle file
    descriptors_dict = pkl.load(f)
    TRAIN_SIZE = len(descriptors_dict["train_images"][1])
descriptors_dict.keys()

#XGBOOST
import xgboost as xgb
import time 

def train_and_get_test_metrics(train_images, train_labels, test_images, test_labels):
    model = xgb.XGBClassifier()

    start_time = time.perf_counter()

    model.fit(train_images, train_labels) #For sklearn no one hot encoding

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.6f} seconds")
    
    #Now predict using the trained RF model. 
    prediction = model.predict(test_images)
    #Print overall accuracy
    #Print overall accuracy
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction))
    f1 = metrics.f1_score(test_labels, prediction)
    print ("F1 = ", f1)
    print ("CM: ")
    print(confusion_matrix(test_labels, prediction))
    return f1

def shuffle_tensors_in_same_order(tensor1, tensor2):
    perm = torch.randperm(len(tensor1))

    shuffled1 = tensor1[perm]
    shuffled2 = tensor2[perm]

    return shuffled1, shuffled2

from sklearn.utils import resample
def getBalancedXy(X, y, replacement=True):
    # Find the indices of positive and negative samples
    pos_indices = (y == 1).nonzero(as_tuple=True)[0]
    neg_indices = (y == 0).nonzero(as_tuple=True)[0]

    if len(pos_indices) == 0 or len(neg_indices) == 0: 
        print("THIS IS NO BUENO !!!!! there are either no pos or neg in this slice")

    # Undersample the majority class or oversample the minority class
    if len(pos_indices) > len(neg_indices):
        # Undersample the positive samples
        neg_indices_upsampled = resample(neg_indices, replace=replacement, n_samples=len(pos_indices))
        indices = torch.cat((pos_indices, neg_indices_upsampled))
    else:
        # Oversample the negative samples
        pos_indices_upsampled = resample(pos_indices, replace=replacement, n_samples=len(neg_indices))
        indices = torch.cat((pos_indices_upsampled, neg_indices))

    # Create a balanced dataset using the selected indices
    balanced_X = X[indices]
    balanced_y = y[indices]

    return balanced_X, balanced_y

def getTrainShuffled_andTestUnshuffled(threshold):
    with open(("data/descriptors_dict_threshold_1_through_10.pkl"), 'rb') as f:
        # Load the object from the pickle file
        descriptors_dict = pkl.load(f)

    # iterate ten times starting with 10% of training data, balancing it and training it, then 20%, then 30%.... Only do it for 2 or 3 thresholds though
    train_labels = descriptors_dict["train_labels"][1]
    train_images = descriptors_dict["train_images"][1]

    train_images, train_labels = shuffle_tensors_in_same_order(train_images, train_labels)

    test_labels = descriptors_dict["test_labels"][1]
    test_images = descriptors_dict["test_images"][1]

    return train_images, train_labels, test_images, test_labels

def getF1sForThreshold(threshold, num_f1s, trainsizes):
    f1_dict_by_trainsize = {}
    for trainsize in trainsizes: 
        f1s = []
        for _ in range(num_f1s):
            train_images, train_labels, test_images, test_labels = getTrainShuffled_andTestUnshuffled(threshold)
            train_images_, train_labels_ = getBalancedXy(train_images[0:trainsize], train_labels[0:trainsize])
            f1 = train_and_get_test_metrics(train_images_, train_labels_, test_images, test_labels)
            f1s.append(f1)

        f1_dict_by_trainsize[trainsize] = f1s
    return f1_dict_by_trainsize

# getting sizes of 20, 40, 60, 80, 100% of training data
trainsizes = []
for i in [2,4,6,8,10]: 
    slice_end = int(TRAIN_SIZE * 0.1*i)
    if slice_end >= TRAIN_SIZE: 
        print("slide_end bigger than train_images len: ", TRAIN_SIZE)
        slice_end = TRAIN_SIZE - 1
    trainsizes.append(slice_end)

threshold1_f1s = getF1sForThreshold(1, 5, trainsizes)
with open('threshold1_f1s.pkl', 'wb') as f:
    pkl.dump(threshold1_f1s, f)



threshold9_f1s = getF1sForThreshold(9, 5, trainsizes)
with open('threshold9_f1s.pkl', 'wb') as f:
    pkl.dump(threshold9_f1s, f)
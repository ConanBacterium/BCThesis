import sys
sys.path.append('../') # add parent dir to path, otherwise can't load ResNet, custom_metric_funcs etc

import torch
import torch.nn as nn # all neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # all optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # all functions that don't have any parameters, relu, sigmoid, softmax, etc.
from torch.utils.data import DataLoader # gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets # has standard datasets we can import in a nice way
import torchvision.transforms as transforms # transform images, videos, etc.
import torchvision.models as models
from resnet import ResNet50
from FungAIDataset import getFungAIDatasetSplits
from preprocessing import resize_224_with_aug_no_norm, resize_224_no_aug_no_norm
from sklearn.metrics import confusion_matrix
import mlflow
from mlflow import pyfunc 
import mlflow.pytorch
import numpy as np
import datetime
from custom_metric_funcs import get_accuracy_and_confusion_matrix

#setting torch hub directory manually
torch.hub.set_dir("/home/anaconda/.cache/torch/hub/")

mlflow.set_experiment("finetuning_pretrained_resnet50_downsizing_to_224x224")

with mlflow.start_run():
    ########### TODO !!!!!!!!! ADD CODE AS ARTIFACT !!!!!!!! AND MAYBE CODE OF HOMEMADE MODULES THAT ARE IMPORTED? 
    ############ AND THE MODEL !!! VERY IMPORTANT! we only save the weights of the model, and thus if the model changes they will become
    ############ useless... 
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyperparamters
    num_classes=1
    learning_rate = 0.001
    batch_size = 20 # tested batch_size of 64, not enough memory for that... 
    num_epochs = 20
    threshold = 0.5
    balanced = True 

    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("threshold", threshold)
    mlflow.log_param("balanced", balanced)

    # load data
    dataset_limit = 0
    testsize = 300
    trainsize = None 
    valsize = 0
    mlflow.log_param("dataset_limit", dataset_limit)
#     mlflow.log_param("valsize", valsize)
    
    mlflow.set_tag("train_preprocessing", "resize_224_with_aug_no_norm")
    mlflow.set_tag("val_preprocessing", "resize_224_no_aug_no_norm")

    train_set, test_set = getFungAIDatasetSplits(valsize, testsize, trainsize=trainsize, train_transform=resize_224_with_aug_no_norm, 
                                                 val_test_transform=resize_224_no_aug_no_norm, limit=dataset_limit, balanced=balanced)
    print(f"testsize {len(test_set)} : trainsize {len(train_set)}")
    mlflow.log_param("testsize", len(test_set))
    mlflow.log_param("trainsize", len(train_set))
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    # initialize network
    model = models.resnet50(pretrained=True)
    for param in model.parameters(): # freeze weights before changing FC-layer
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(2048, num_classes), # these weights aren't frozen
                              nn.Sigmoid()) 
    model.to(device)

    # loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train network
    model.train() # set model to training mode 
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader): #data is image, target is true y (true class)
            data = data.to(device=device)
            targets = targets.to(device=device)
            targets = targets.float()
            targets = targets.view(-1, 1) # Reshape target tensors to match output tensor size... TODO CHECK IF THIS IS BUENO

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad() # set all gradients to 0 before starting to do backpropragation for eatch batch because gradients are accumulated
            loss.backward()

            # gradient descent or adam step
            optimizer.step() # update the weights
            
        accuracy, cm = get_accuracy_and_confusion_matrix(test_loader, model, device, threshold)
        print(f"epoch {epoch} CM")
        print(cm)
        print(f"accuracy {accuracy}")
        print("---")
        mlflow.log_metric("accuracy", accuracy, step=epoch)
        mlflow.log_metric("tp", cm[0][0], step=epoch)
        mlflow.log_metric("fp", cm[0][1], step=epoch)
        mlflow.log_metric("tn", cm[1][1], step=epoch)
        mlflow.log_metric("fn", cm[1][0], step=epoch)

    # Define the file path to save the weights
    state_dict_path = '../resnet_weights/finetuned_pretrained_resnet50_downsizing_to_224x224.pth'
    # Save the weights
    torch.save(model.state_dict(), state_dict_path)

    mlflow.pytorch.log_model(model, "models")

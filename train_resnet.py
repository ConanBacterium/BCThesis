import torch
import torch.nn as nn # all neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # all optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # all functions that don't have any parameters, relu, sigmoid, softmax, etc.
from torch.utils.data import DataLoader # gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets # has standard datasets we can import in a nice way
import torchvision.transforms as transforms # transform images, videos, etc.
from resnet import ResNet50
from FungAIDataset import FungAIDataset
from preprocessing import fungai_preprocessing
from sklearn.metrics import confusion_matrix
import mlflow
from mlflow import pyfunc 
import mlflow.pytorch
import numpy as np
import datetime
from custom_metric_funcs import get_accuracy_and_confusion_matrix

mlflow.set_experiment("balanced_resnet50_5epochs")

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
    num_epochs = 5
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
    mlflow.set_tag("dataset_limit", dataset_limit)
    dataset = FungAIDataset(transform = fungai_preprocessing, limit=dataset_limit, balanced=balanced)

    mlflow.set_tag("preprocessing", "fungai_preprocessing")

    testsize = 300
    trainsize = len(dataset)-testsize
    mlflow.set_tag("trainsize", trainsize)
    mlflow.set_tag("testsize", testsize)
    train_set, test_set = torch.utils.data.random_split(dataset, [trainsize, testsize])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    # initialize network
    model = ResNet50(num_classes=num_classes).to(device)

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
        mlflow.log_metric("accuracy", accuracy, step=epoch)
        mlflow.log_metric("tp", cm[0][0], step=epoch)
        mlflow.log_metric("fp", cm[0][1], step=epoch)
        mlflow.log_metric("tn", cm[1][1], step=epoch)
        mlflow.log_metric("fn", cm[1][0], step=epoch)

    # Define the file path to save the weights
    state_dict_path = 'resnet_weights/resnet50_10epochs.pth'
    # Save the weights
    torch.save(model.state_dict(), state_dict_path)

    mlflow.pytorch.log_model(model, "models")

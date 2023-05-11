import sys
sys.path.append('../') # add parent dir to path, otherwise can't load ResNet, custom_metric_funcs etc

import torch
import torch.nn as nn # all neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # all optimization algorithms, SGD, Adam, etc.
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F # all functions that don't have any parameters, relu, sigmoid, softmax, etc.
from torch.utils.data import DataLoader # gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets # has standard datasets we can import in a nice way
import torchvision.transforms as transforms # transform images, videos, etc.
import torchvision.models as models
from resnet import ResNet50
from FungAIDataset import getFungAIDatasetSplits
from preprocessing import fungai_preprocessing_resize_to_224_w_augmentation, fungai_preprocessing_resize_to_224_without_augmentation
from sklearn.metrics import confusion_matrix
import mlflow
from mlflow import pyfunc 
import mlflow.pytorch
import numpy as np
import datetime
from custom_metric_funcs import get_accuracy_and_confusion_matrix, get_threshold_optimized_for_f1
from train_early_stopping_and_lr_scheduler import train_early_stopping_lr_scheduler
from mlflow_custom_utils import mlflow_log_files_in_dir
from graph_generators import generate_val_train_loss_graphs
from FC_for_effnet import EffnetDenseNet

#setting torch hub directory manually
torch.hub.set_dir("/home/anaconda/.cache/torch/hub/")

mlflow.set_experiment("resnet50_224x225_denseclassifier")

with mlflow.start_run():
    mlflow_run_id = mlflow.active_run().info.run_id
    
    mlflow_log_files_in_dir("../", mlflow)
    mlflow.log_param("executed_script", __file__)
    
    log_path = f"../run_logs/{mlflow_run_id}.txt"
    with open(log_path, 'w') as file:
        # Append a string to the file with a newline character
        file.write("Log for {mlflow_run_id}.\n")
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyperparamters
    num_classes=1
    learning_rate = 0.0005
    batch_size = 20 # tested batch_size of 64, not enough memory for that... 
    num_epochs = 2
    threshold = 0.5
    balanced = True 
    patience = 11
    lr_stepsize = 5
    lr_gamma = 0.1 # reduces by factor of 0.1 every lr_stepsize epochs
    decay_rate = 0.001
    

    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("threshold", threshold)
    mlflow.log_param("balanced", balanced)
    mlflow.log_param("patience", patience)
    mlflow.log_param("lr_stepsize", lr_stepsize)
    mlflow.log_param("lr_gamma", lr_gamma)
    mlflow.log_param("decay_rate", decay_rate)

    # load data
    dataset_limit = 40
    testsize = 20
    trainsize = None 
    valsize = 0
    mlflow.log_param("dataset_limit", dataset_limit)
#     mlflow.log_param("valsize", valsize)
    
    mlflow.set_tag("train_preprocessing", "fungai_preprocessing_resize_to_224_w_augmentation")
    mlflow.set_tag("val_preprocessing", "fungai_preprocessing_resize_to_224_without_augmentation")

    train_set, test_set = getFungAIDatasetSplits(valsize, testsize, trainsize=trainsize, train_transform=fungai_preprocessing_resize_to_224_w_augmentation,                                  val_test_transform=fungai_preprocessing_resize_to_224_without_augmentation, limit=dataset_limit, balanced=balanced)
    print(f"testsize {len(test_set)} : trainsize {len(train_set)}")
    mlflow.log_param("testsize", len(test_set))
    mlflow.log_param("trainsize", len(train_set))
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    # initialize network
    model = models.resnet50(pretrained=True)
    for param in model.parameters(): # freeze weights before changing FC-layer
        param.requires_grad = False
#     model.fc = nn.Sequential(nn.Linear(2048, num_classes), # these weights aren't frozen
#                               nn.Sigmoid()) 
    model.fc = nn.Sequential(nn.Linear(2048, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.65),
                                      nn.Linear(1024, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.65),
                                      nn.Linear(512, 1),
                                      nn.Sigmoid()) 
    model.to(device)

    # loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=lr_gamma)

    # train network
    train_losses, val_losses, val_accs, cms, best_val_acc, last_epoch, best_epoch = train_early_stopping_lr_scheduler(model, train_loader, test_loader, optimizer, criterion, num_epochs, patience, scheduler, device, threshold,  mlflow_run_id, "../model_weights/resnet50_224x224.pth", log_path)
    
    mlflow.log_artifact(f"../run_logs/{mlflow_run_id}.txt")
    mlflow.set_tag("last_epoch", last_epoch)
    # save val_train_graph
    val_train_loss_graph_path = f"../graphs/{mlflow_run_id}.png"
    generate_val_train_loss_graphs(train_losses, val_losses, val_train_loss_graph_path)
    mlflow.log_artifact(val_train_loss_graph_path)
    # log metrics
#     mlflow.log_metric("accuracy", val_accs[-1], step=num_epochs)
    mlflow.log_metric("best_val_acc", best_val_acc)
    mlflow.log_metric("tp", cms[best_epoch][1][1], step=best_epoch)
    mlflow.log_metric("fp", cms[best_epoch][0][1], step=best_epoch)
    mlflow.log_metric("tn", cms[best_epoch][0][0], step=best_epoch)
    mlflow.log_metric("fn", cms[best_epoch][1][0], step=best_epoch)
    print(f"Best epoch was {best_epoch} with best_val_acc {best_val_acc}")
    
    print("best_threshold, best_metric_score")
    print(get_threshold_optimized_for_f1(test_loader, model, device))

#     # Save a checkpoint after every epoch
    checkpoint_path = f"../model_weights/resnet50_224x224_checkpoint_{last_epoch}.pt"
    torch.save({
        'epoch': last_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1]
    }, checkpoint_path)

    mlflow.pytorch.log_model(model, "models")

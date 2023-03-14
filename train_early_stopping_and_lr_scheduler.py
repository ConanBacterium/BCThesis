import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix

def print_and_log(string, filename):
    with open(filename, 'a') as file:
        file.write(string)
        file.write("\n")
    print(string)

def train_early_stopping_lr_scheduler(model, train_loader, val_loader, optimizer, criterion, n_epochs, patience, lr_scheduler, device, threshold, run_id, weight_path):
    log_path = f"../run_logs/{run_id}.txt"
    with open(log_path, 'w') as file:
        # Append a string to the file with a newline character
        file.write("Log for {run_id}.\n")
    
    train_loss = []
    val_loss = []
    val_acc = []
    cm = []

    best_val_acc = 0.0
    epochs_since_improvement = 0
    
    model.train() # set model to training mode 
    for epoch in range(n_epochs):
        print_and_log(f"Training epoch {epoch}", log_path)
        epoch_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader): #data is image, target is true y (true class)
            data = data.to(device=device)
            targets = targets.to(device=device)
            targets = targets.float()
            targets = targets.view(-1, 1) # Reshape target tensors to match output tensor size... TODO put this in Dataset.__getitem__ method??

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad() # set all gradients to 0 before starting to do backpropragation for eatch batch because gradients are accumulated
            loss.backward()
            
            epoch_loss += loss.item()
            
            optimizer.step() # update the weights
    
    
        train_loss.append(epoch_loss / len(train_loader))
        print_and_log(f"train loss: {train_loss}", log_path)
        # Evaluate the model on the validation data
        model.eval()
        epoch_loss = 0.0
        epoch_true = []
        epoch_pred = []
        epoch_num_correct = 0
        epoch_num_samples = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device=device)
                y = y.to(device=device) # unnecessary, it's just a 0 or 1... 
                y = y.float()
                y = y.view(-1,1)
                
                scores = model(x)
                predictions = torch.round(scores - threshold + 0.5)
                
                y = y.cpu()
                predictions = predictions.cpu()
                scores = scores.cpu()
                
                epoch_true.extend(y)
                epoch_pred.extend(predictions)
                
                epoch_num_correct += (predictions == y).sum() # sum because the dataloader might be in batches.
                epoch_num_samples += predictions.size(0)
                
                loss = criterion(y, scores) # not thresholded, this is to keep track of epoch loss, not validation
                epoch_loss += loss.item() # get scalar value of loss function 
                
            y_true = [label.item() for label in epoch_true]
            y_pred = [label.item() for label in epoch_pred] # THIS PRODUCES NAN... 
            cm.append(confusion_matrix(y_true, y_pred))
        
            val_loss.append(epoch_loss / len(val_loader))
        
            accuracy = float(epoch_num_correct)/float(epoch_num_samples)*100
            val_acc.append(accuracy)
            print_and_log(f"epoch {epoch} with validation accuracy {val_acc[-1]} and CM", log_path)
            print_and_log(np.array2string(cm[-1]), log_path)

            # Check if the validation accuracy has improved
            #### TODO use other metric 
            if val_acc[-1] > best_val_acc:
                best_val_acc = val_acc[-1]
                epochs_since_improvement = 0
               
                # Save the weights
                torch.save(model.state_dict(), weight_path)
            else:
                epochs_since_improvement += 1

                if epochs_since_improvement == patience // 2:
                    lr_scheduler.step()

            if epochs_since_improvement >= patience:
                print_and_log('Stopping training after {} epochs without improvement.'.format(patience), log_path)
                break
    return train_loss, val_loss, val_acc, cm, best_val_acc, epoch

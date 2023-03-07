import torch
from sklearn.metrics import confusion_matrix

def get_accuracy_and_confusion_matrix(loader, model, device, threshold):
    num_correct = 0
    num_samples = 0
    model.eval() # set model to evaluation mode

    y_true = []
    y_pred = []

    with torch.no_grad(): # don't need to calculate gradients
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device) # unnecessary, it's just a 0 or 1... 
            y = y.float()
            y = y.view(-1,1)

            scores = model(x)
            predictions = torch.round(scores - threshold + 0.5)
            
            y = y.cpu()
            predictions = predictions.cpu()

            y_true.extend(y)
            y_pred.extend(predictions)

            num_correct += (predictions == y).sum() # sum because the dataloader might be in batches.
            num_samples += predictions.size(0)

        accuracy = float(num_correct)/float(num_samples)*100

#         print(f"shape of y_true[0]: {y_true[0].shape}, shape of y_pred[0]: {y_pred[0].shape}")
#         print(f"y_true uniques: {np.unique(y_true)}, y_pred uniques: {np.unique(y_pred)}")
        
        y_true = [label.item() for label in y_true]
        y_pred = [label.item() for label in y_pred]
        cm = confusion_matrix(y_true, y_pred)
        
    model.train() # set model back to training mode
    return (accuracy, cm)

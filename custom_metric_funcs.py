import torch
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
import math

def calcRecall(TP, FN): return TP / (TP+FN +1e-9) # add small number to avoid possible division by zero 
def calcPrecision(TP, FP): return TP / (TP+FP +1e-9)


def f1(cm):
    tp = cm[1][1]
    fp = cm[0][1]
    tn = cm[0][0]
    fn = cm[1][0]
    precision = calcPrecision(tp, fp)
    recall = calcRecall(tp, fn)
    return 2 * ( (precision * recall) / (precision + recall +1e-9) ) # add small number to avoid possible division by zero 

# matthew's correlation coefficient (phi coefficient)
def mmc(cm):
    TP = cm[1][1]
    FP = cm[0][1]
    TN = cm[0][0]
    FN = cm[1][0]
    return ( (TP * TN) - (FP * FN) ) / ( math.sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) ))

def calcRecallFromCM(cm):
    tp = cm[1][1]
    fn = cm[1][0]
    recall = calcRecall(tp, fn)
    return recall

def threshold_optimized_for_metricfunc(y_true, y_probabilities, metricfunc):
    """
    metricfunc takes confusion matrix and returns metric score
    """
    candidate_thresholds = np.linspace(0, 0.95, 150)
    best_threshold = 0.5
    best_metric_score = 0
    best_cm = None
    
    for candidate_threshold in candidate_thresholds:
        y_preds = [1.0 if res >= candidate_threshold else 0.0 for res in y_probabilities]
        # TODO  calculate CM instead and then pass that to metricfunc so we can get full information ... !!! 
        cm = confusion_matrix(y_true, y_preds) 
        metric = metricfunc(cm)
        
        if metric > best_metric_score: 
            best_metric_score = metric
            best_threshold = candidate_threshold
            best_cm = cm
            
            
    return best_threshold, best_metric_score, best_cm

def get_ytrue_and_yprobabilities(loader, model, device):
    y_true = []
    y_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device) # unnecessary, it's just a 0 or 1... 
            y = y.float()
            y = y.view(-1,1)
            scores = model(x)
            
            y = y.cpu().detach()
            scores = scores.cpu().detach()

            y_true.extend(y.numpy())
            y_probabilities.extend(scores.numpy())
    return y_true, y_probabilities
    
def get_threshold_optimized_for_f1(loader, model, device):
    y_true, y_probabilities = get_ytrue_and_yprobabilities(loader, model, device)
    return threshold_optimized_for_metricfunc(y_true, y_probabilities, f1)

def get_threshold_optimized_for_recall(loader, model, device):
    y_true, y_probabilities = get_ytrue_and_yprobabilities(loader, model, device)
    return threshold_optimized_for_metricfunc(y_true, y_probabilities, calcRecallFromCM)

def get_accuracy_and_confusion_matrix(loader, model, device, threshold):
    # TODO put model.eval()
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

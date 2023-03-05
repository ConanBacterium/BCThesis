import torch
import torch.nn as nn # all neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # all optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # all functions that don't have any parameters, relu, sigmoid, softmax, etc.
from torch.utils.data import DataLoader # gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets # has standard datasets we can import in a nice way
import torchvision.transforms as transforms # transform images, videos, etc.
from resnet import ResNet50
from FungAIDataset import FungAIDataset

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparamters
num_classes=1
learning_rate = 0.001
batch_size = 5
num_epochs = 1
threshold = 0.5

# load data
dataset = FungAIDataset(transform = transforms.ToTensor(), limit = 3000) 

testsize = 600
train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset)-testsize, testsize])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# initialize network
model = ResNet50(num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
model.train() # set model to training mode 
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader): #data is image, target is true y (true class)
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad() # set all gradients to 0 before starting to do backpropragation for eatch batch because gradients are accumulated
        loss.backward()

        # gradient descent or adam step
        optimizer.step() # update the weights

# check accuracy on training & test

print("!!!!!")
print(scores)
print(scores.shape)

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')
    num_correct = 0
    num_samples = 0
    model.eval() # set model to evaluation mode

    with torch.no_grad(): # don't need to calculate gradients
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            predictions = torch.round(scores - threshold + 0.5)
            num_correct += (predictions == y).sum() # sum because the dataloader might be in batches.
            num_samples += predictions.size(0)

        print(f'{num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train() # set model back to training mode

check_accuracy(test_loader, model)

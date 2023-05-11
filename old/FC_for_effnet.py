import torch
import torch.nn as nn

class EffnetDenseNet(nn.Module):
    def __init__(self,
                 input_size = 2560 ,
                 fc1_size = 1280,
                 fc2_size = 630,
                 fc3_size = 120,
                 fc1_dropout = 0.65,
                 fc2_dropout = 0.65,
                 fc3_dropout = 0.65,
                 output_dim = 1):
        super(EffnetDenseNet, self).__init__()
        
        self.input_size = input_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.fc3_size = fc3_size
        self.fc1_dropout = fc1_dropout
        self.fc2_dropout = fc2_dropout
        self.fc3_dropout = fc3_dropout

        self.output_dim = output_dim 

        self.ff_model = nn.Sequential(nn.Linear(input_size, fc1_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc1_dropout),
                                      nn.Linear(fc1_size, fc2_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc2_dropout),
                                      nn.Linear(fc2_size, fc3_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc3_dropout),
                                      nn.Linear(fc3_size, output_dim),
                                      nn.Sigmoid())
        
    
    def forward(self, x):
        output = self.ff_model(x)                                               
        return output

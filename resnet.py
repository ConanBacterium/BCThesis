import torch 
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        # identity_downsample is a function (transform) for the skipconnections... if in_channels or dimension has changed, we skipped connection must be resized also
        super(Block, self).__init__()
        self.expansion = 4 # outchannels = 4*inchannels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) # same convolution??
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1) # 
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0) #
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample # if we have a skip connection, we need to resize it to the new dimension. there is a skip connection if the in_channels or dimension has changed. (after each block or what??)

    def forward(self, x):
        identity = x # save the input for the skip connection... 

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        x += identity # add the skip connection
        x = self.relu(x)
        
        return x
    
class BlockSmall(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(BlockSmall, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) # same convolution??
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1) # 
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample # if we have a skip connection, we need to resize it to the new dimension. there is a skip connection if the in_channels or dimension has changed. (after each block or what??)

    def forward(self, x):
        identity = x # save the input for the skip connection... 
        print(f"identity.size {identity.size()}")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            print("identity_downsample not none")
            identity = self.identity_downsample(identity)
            print(f"identity.size {identity.size()}")
        print(f"x.size {x.size()}")
        x += identity # add the skip connection
        x = self.relu(x)
        
        return x
    
class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        # layers is a list of the number of blocks in each layer, so ResNet50 is [3,4,6,3]
        super(ResNet, self).__init__()
        self.in_channels = 64 # channels in first layer
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3) # 224x224 -> 112x112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 112x112 -> 56x56
        self.sigmoid = nn.Sigmoid()

        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)  # # 112x112 -> 56x56 happens in maxpool, which is considered part of this layer.? 
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2) # 56x56 -> 28x28
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride = 2) # 28x28 -> 14x14
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride = 2) # 14x14 -> 7x7 # and 2048 channels

        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) 
        self.fc = nn.Linear(512*4, num_classes) # 512*4, last out_channel * expansion

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) # flatten
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
        
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        # num_residual_blocks is the size of the res blocks in each layer 3,4,6,3 for ResNet50
        identity_downsample = None
        layers = []

        # note that the stride of the conv is equal to stride, and condition requires that it's more than 1. So this resizes the output.
        # NOTE!! This is no bueno for BlockSmall... 
        if stride != 1 or self.in_channels != out_channels * 4: 
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                                                nn.BatchNorm2d(out_channels*4))
        
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4  

        for i in range(num_residual_blocks - 1): # -1 because we already have one block
            layers.append(block(self.in_channels, out_channels)) # 256 -> 64, 64*4 -> 256

        return nn.Sequential(*layers)
    
def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(Block, [3,4,6,3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(Block, [3,4,23,3], img_channels, num_classes)

def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(Block, [3,4,36,3], img_channels, num_classes)

def ResNet18(img_channels=3, num_classes=1000):
    return ResNet(BlockSmall, [2,2,2,2], img_channels, num_classes)

""" def test():
    x = torch.randn(2, 3, 600, 600)
    model = ResNet50()
    y = model(x).to('cpu')
    print(y.shape)

test() """

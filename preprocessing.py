from torchvision import transforms
import torch 

fungai_preprocessing = transforms.Compose([
    transforms.Resize(600), transforms.CenterCrop(600), transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5), transforms.RandomCrop(600,padding=67), transforms.ToTensor(), 
    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=.5, hue=.3),]), p=0.3), 
    transforms.RandomRotation(180), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    transforms.RandomErasing()])
